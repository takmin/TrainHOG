#include "FileIO.h"
#include <fstream>

//! CSVファイルをstd::vectorとして読み込み
bool ReadList(const std::string& filename, std::vector<std::string>& dst_vector)
{
	std::ifstream ifs(filename.c_str());
	if (!ifs.is_open())
		return false;

	std::string buf;
	while (ifs && std::getline(ifs, buf)) {
		dst_vector.push_back(buf);
	}

	return true;
}

// input_stringをseparaterで分離
std::vector<std::string> TokenizeString(const std::string& input_string, const std::vector<std::string>& separater_vec)
{
	if (input_string.empty())
		return std::vector<std::string>();

	std::vector<std::string>::const_iterator separater_itr;
	std::vector<std::string::size_type>	index_vec;
	std::string::size_type	index;
	for (separater_itr = separater_vec.begin(); separater_itr != separater_vec.end(); separater_itr++) {
		index = 0;
		while (true) {
			index = input_string.find(*separater_itr, index);
			if (index == std::string::npos) {
				break;
			}
			else {
				index_vec.push_back(index);
				index++;
			}
		}
	}
	sort(index_vec.begin(), index_vec.end());

	std::vector<std::string> ret_substr_vec;
	std::vector<std::string::size_type>::iterator idx_itr;
	std::string::size_type start_idx = 0;
	int str_size;
	for (idx_itr = index_vec.begin(); idx_itr != index_vec.end(); idx_itr++) {
		str_size = *idx_itr - start_idx;
		ret_substr_vec.push_back(input_string.substr(start_idx, str_size));
		start_idx = *idx_itr + 1;
	}
	ret_substr_vec.push_back(input_string.substr(start_idx));

	return ret_substr_vec;
}


// CSVファイルからストリングリストを取得
bool ReadCSVFile(const std::string& input_file, std::vector < std::vector <std::string>> &output_strings,
	const std::vector<std::string>& separater_vec)
{
	std::vector<std::string> sep_vec;
	if (separater_vec.empty()) {
		sep_vec.push_back(",");
	}
	else {
		sep_vec = separater_vec;
	}
	std::ifstream ifs(input_file.c_str());
	if (!ifs.is_open())
		return false;

	output_strings.clear();

	std::string buf;
	while (ifs && std::getline(ifs, buf)) {
		std::vector<std::string> str_list = TokenizeString(buf, sep_vec);
		output_strings.push_back(str_list);
	}
	return true;
}


//! アノテーションファイルの読み込み
/*!
opencv_createsamles.exeと同形式のアノテーションファイル読み書き
ReadCsvFile()関数必須
\param[in] gt_file アノテーションファイル名
\param[out] imgpathlist 画像ファイルへのパス
\param[out] rectlist 各画像につけられたアノテーションのリスト
\return 読み込みの成否
*/
bool LoadAnnotationFile(const std::string& gt_file, std::vector<std::string>& imgpathlist, std::vector<std::vector<cv::Rect>>& rectlist)
{
	std::vector<std::vector<std::string>> tokenized_strings;
	std::vector<std::string> sep;
	sep.push_back(" ");
	if (!ReadCSVFile(gt_file, tokenized_strings, sep))
		return false;

	std::vector<std::vector<std::string>>::iterator it, it_end = tokenized_strings.end();
	for (it = tokenized_strings.begin(); it != it_end; it++) {
		int num_str = it->size();
		if (num_str < 2)
			continue;

		std::string filename = (*it)[0];
		if (filename.empty() || filename.find("#") != std::string::npos) {
			continue;
		}

		imgpathlist.push_back(filename);
		int obj_num = atoi((*it)[1].c_str());
		std::vector<cv::Rect> rects;
		for (int i = 0; i<obj_num && 4 * i + 6 <= num_str; i++) {
			int j = 4 * i + 2;
			cv::Rect obj_rect;
			obj_rect.x = atoi((*it)[j].c_str());
			obj_rect.y = atoi((*it)[j + 1].c_str());
			obj_rect.width = atoi((*it)[j + 2].c_str());
			obj_rect.height = atoi((*it)[j + 3].c_str());
			rects.push_back(obj_rect);
		}
		rectlist.push_back(rects);
	}

	return true;
}

