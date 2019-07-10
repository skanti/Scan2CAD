#pragma once

// Solution dir should be set to the absolute path of the solution root folder. We set a default value just
// for correct code highlighting.
#ifndef SOLUTION_DIR
#define SOLUTION_DIR ""
#endif

#define DATA_DIR SOLUTION_DIR "data\\"
#define OUTPUT_DIR SOLUTION_DIR "output\\"

#define CONCAT_STR(a,b) std::string(a) + std::string(b)
#define DATA(a) CONCAT_STR(DATA_DIR, a)
#define OUTPUT(a) CONCAT_STR(OUTPUT_DIR, a)