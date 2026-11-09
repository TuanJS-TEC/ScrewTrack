#pragma once
#include <ostream>
#include <string>
#include <vector>

#include "types.h"

// Header "session,image,side,status,rotation_deg,reason"; rotation with 2 decimals, empty when unmeasurable.
// Reason strings contain no commas, so fields are not quoted.
void writeCsv(std::ostream& out, const std::string& sessionName, const std::vector<NutResult>& rows);
