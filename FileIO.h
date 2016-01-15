#ifndef __FILEIO__
#define __FILEIO__

#include <opencv2/core.hpp>

bool ReadList(const std::string& filename, std::vector<std::string>& dst_vector);
bool LoadAnnotationFile(const std::string& gt_file, std::vector<std::string>& imgpathlist, std::vector<std::vector<cv::Rect>>& rectlist);

#endif
