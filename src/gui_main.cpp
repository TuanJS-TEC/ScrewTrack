#include <QApplication>
#include <QComboBox>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileInfo>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QMimeData>
#include <QPushButton>
#include <QThread>
#include <QVBoxLayout>
#include <atomic>
#include <map>
#include <opencv2/imgproc.hpp>

#include "trackrun.h"

// Runs one video on its own thread; the window only shows what it emits.
class Job : public QThread {
    Q_OBJECT
public:
    Job(QString model, QString db, QString video, bool reg, QObject* parent)
        : QThread(parent), model_(model), db_(db), video_(video), reg_(reg) {}
    void stop() { stop_ = true; }

signals:
    void frame(QImage img, QString info);
    void done(QString msg);

protected:
    void run() override {
        try {
            const NutDetector det(model_.toStdString());
            NutDb db(db_.toStdString());
            const Registry reg = reg_ ? Registry() : db.load();
            std::set<int> registered;
            for (const RegNut& n : reg.nuts) registered.insert(n.id);
            std::map<int, NutResult> turned;  // last measurement per id

            const TrackRun r = runSource(det, video_.toStdString(), [&](int f, const cv::Mat& img, const std::vector<TrackedBox>& tracks) {
                const auto rot = measureRotations(reg, img, tracks);
                cv::Mat vis = img.clone();
                const int thick = std::max(2, img.cols / 400);
                QStringList lines;
                for (const TrackedBox& t : tracks) {
                    const auto m = rot.find(t.id);
                    if (m != rot.end()) turned.insert_or_assign(t.id, m->second);
                    // register: yellow. track: red = no such id in the database; otherwise by the measurement:
                    // green ok, orange loose, blue not measurable (or no measurement, e.g. video)
                    cv::Scalar c(0, 220, 255);
                    std::string label = std::to_string(t.id);
                    if (!reg_) {
                        c = !registered.count(t.id) ? cv::Scalar(0, 0, 255) : m == rot.end() ? cv::Scalar(0, 200, 0)
                            : m->second.status == Status::Ok ? cv::Scalar(0, 200, 0)
                            : m->second.status == Status::Loose ? cv::Scalar(0, 140, 255) : cv::Scalar(255, 160, 0);
                        if (m != rot.end() && m->second.rotationDeg) {
                            char deg[32];
                            std::snprintf(deg, sizeof deg, "  %.1f", *m->second.rotationDeg);
                            label += deg;
                        }
                        if (m != rot.end())
                            lines << QString("ID %1: %2").arg(t.id).arg(m->second.rotationDeg
                                         ? QString("%1° (%2)").arg(*m->second.rotationDeg, 0, 'f', 2).arg(m->second.status == Status::Loose ? "lỏng" : "ok")
                                         : QString("không đo được (%1)").arg(QString::fromStdString(m->second.reason)));
                        else if (!registered.count(t.id)) lines << QString("ID %1: chưa đăng ký").arg(t.id);
                    }
                    cv::rectangle(vis, t.box, c, thick);
                    cv::putText(vis, label, t.box.tl() + cv::Point(0, -thick * 3), cv::FONT_HERSHEY_SIMPLEX, img.cols / 800.0, c, thick);
                }
                cv::cvtColor(vis, vis, cv::COLOR_BGR2RGB);
                emit frame(QImage(vis.data, vis.cols, vis.rows, int(vis.step), QImage::Format_RGB888).copy(),
                           QString("#%1 · %2 ốc").arg(f).arg(tracks.size()) + (lines.isEmpty() ? "" : " · " + lines.join(" · ")));
                return !stop_;
            }, reg);

            if (stop_) return emit done("Đã dừng. Chưa ghi gì vào database.");
            if (reg_) {
                Registry out{{}, r.photos};
                for (const auto& [id, n] : r.first) out.nuts.push_back(n);
                db.save(out);
                emit done(QString("Xong: đã đăng ký %1 ốc vào database.").arg(out.nuts.size()));
            } else {
                QStringList unknown, missing, loose, unmeasured;
                for (int id : r.seen)
                    if (!registered.count(id)) unknown << QString::number(id);
                for (int id : registered)
                    if (!r.seen.count(id)) missing << QString::number(id);
                for (const auto& [id, n] : turned)
                    if (n.status == Status::Loose) loose << QString::number(id);
                    else if (n.status == Status::Unmeasurable) unmeasured << QString::number(id);
                emit done(QString("Xong: thấy %1 ốc. Chưa đăng ký: %2. Không thấy: %3. Lỏng (≥%4°): %5. Không đo được: %6.")
                              .arg(r.seen.size()).arg(unknown.join(", ")).arg(missing.join(", ")).arg(kLooseDeg)
                              .arg(loose.join(", ")).arg(unmeasured.join(", ")));
            }
        } catch (const std::exception& e) {
            emit done(QString("Lỗi: ") + e.what());
        }
    }

private:
    QString model_, db_, video_;
    bool reg_;
    std::atomic<bool> stop_{false};
};

class Window : public QWidget {
    Q_OBJECT
public:
    explicit Window(const QString& video) {
        setWindowTitle("ScrewTrack");
        setAcceptDrops(true);
        video_ = new QLineEdit(video);
        model_ = new QLineEdit(PROJECT_DIR "/model/best.onnx");
        db_ = new QLineEdit(PROJECT_DIR "/nuts.db");
        mode_ = new QComboBox;
        mode_->addItem("Đăng ký lần đầu (register)", true);
        mode_->addItem("Đo lại, nhận ID cũ (track)", false);
        run_ = new QPushButton("Chạy");
        view_ = new QLabel("Chọn hoặc kéo thả video, ảnh, hoặc thư mục ảnh vào cửa sổ, rồi bấm Chạy");
        view_->setAlignment(Qt::AlignCenter);
        view_->setMinimumSize(640, 360);
        view_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
        view_->setStyleSheet("background:#111;color:#aaa");
        info_ = new QLabel;
        info_->setWordWrap(true);

        auto* lay = new QVBoxLayout(this);
        lay->addLayout(row("Video / ảnh", video_, "Video, ảnh (*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png);;Tất cả (*)", false, true));
        lay->addLayout(row("Model", model_, "ONNX (*.onnx)", false, false));
        lay->addLayout(row("Database", db_, "SQLite (*.db *.sqlite)", true, false));
        auto* ctl = new QHBoxLayout;
        ctl->addWidget(mode_, 1);
        ctl->addWidget(run_);
        lay->addLayout(ctl);
        lay->addWidget(view_, 1);
        lay->addWidget(info_);
        connect(run_, &QPushButton::clicked, this, &Window::toggle);
    }
    ~Window() override {
        if (job_) {
            job_->stop();
            job_->wait();
        }
    }

protected:
    void dragEnterEvent(QDragEnterEvent* e) override {
        if (e->mimeData()->hasUrls()) e->acceptProposedAction();
    }
    void dropEvent(QDropEvent* e) override { video_->setText(e->mimeData()->urls().value(0).toLocalFile()); }

private:
    QLayout* row(const QString& name, QLineEdit* edit, const QString& filter, bool save, bool folder) {
        auto* h = new QHBoxLayout;
        auto* b = new QPushButton("Chọn…");
        auto* l = new QLabel(name);
        l->setMinimumWidth(70);
        h->addWidget(l);
        h->addWidget(edit, 1);
        h->addWidget(b);
        if (folder) {  // a folder of photos
            auto* d = new QPushButton("Thư mục…");
            h->addWidget(d);
            connect(d, &QPushButton::clicked, this, [=] {
                const QString p = QFileDialog::getExistingDirectory(this, name, edit->text());
                if (!p.isEmpty()) edit->setText(p);
            });
        }
        connect(b, &QPushButton::clicked, this, [=] {
            const QString p = save ? QFileDialog::getSaveFileName(this, name, edit->text(), filter, nullptr, QFileDialog::DontConfirmOverwrite)
                                   : QFileDialog::getOpenFileName(this, name, edit->text(), filter);
            if (!p.isEmpty()) edit->setText(p);
        });
        return h;
    }

    void toggle() {
        if (job_) return job_->stop();
        const bool reg = mode_->currentData().toBool();
        for (QLineEdit* e : {video_, model_})
            if (!QFileInfo::exists(e->text())) return void(info_->setText("Không thấy file: " + e->text()));
        if (reg && QFileInfo::exists(db_->text())) {
            try {
                NutDb db(db_->text().toStdString());
                bool hasData;
                try {
                    hasData = !db.load().nuts.empty();
                } catch (const std::exception&) {  // old format: registering replaces it, so it is still worth asking
                    hasData = true;
                }
                if (hasData && QMessageBox::question(this, "Ghi đè?", "Database đã có ốc đăng ký. Đăng ký lại sẽ xoá chúng. Tiếp tục?") != QMessageBox::Yes)
                    return;
            } catch (const std::exception& e) {
                return info_->setText(QString("Lỗi: ") + e.what());
            }
        }
        job_ = new Job(model_->text(), db_->text(), video_->text(), reg, this);
        connect(job_, &Job::frame, this, [this](QImage img, QString info) {
            view_->setPixmap(QPixmap::fromImage(img).scaled(view_->size(), Qt::KeepAspectRatio, Qt::FastTransformation));
            info_->setText(info);
        });
        connect(job_, &Job::done, this, [this](QString msg) { info_->setText(msg); });
        connect(job_, &QThread::finished, this, [this] {
            job_->deleteLater();
            job_ = nullptr;
            run_->setText("Chạy");
        });
        run_->setText("Dừng");
        info_->setText("Đang nạp model…");
        job_->start();
    }

    QLineEdit *video_, *model_, *db_;
    QComboBox* mode_;
    QPushButton* run_;
    QLabel *view_, *info_;
    Job* job_ = nullptr;
};

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    Window w(argc > 1 ? argv[1] : "");
    w.resize(1000, 720);
    w.show();
    return app.exec();
}

#include "gui_main.moc"
