#ifndef __OUTPUT_H
#define __OUTPUT_H

#include <map>
#include <string>
#include "data.h"

namespace Output {
  void saveVTI(std::map<std::string,Field*> fields, FieldB &fixed);
}

#endif
