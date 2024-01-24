#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>

/**
 * string split
 * @param str is the std::string that is to be splitted
 * @param split is the separator to split str
 * @return std::vector<T> output
 */
std::vector<std::string> Stringsplit(std::string str, const char split = ',');

template <typename T>
std::vector<T> Stringsplit(std::string str, const char split = ',')
{
	bool error = false;
	std::string token;					 // recive buffers
	std::vector<T> output;				 // recive buffers
	std::istringstream iss(str);		 // input stream
	std::vector<std::string> str_output; // recive buffers

	while (getline(iss, token, split)) // take "split" as separator
	{
		str_output.push_back(token);
	}
	if (typeid(T) == typeid(int))
		std::transform(str_output.begin(), str_output.end(), std::back_inserter(output),
					   [](std::string &sr)
					   { return std::stoi(sr); });
	else if (typeid(T) == typeid(float))
		std::transform(str_output.begin(), str_output.end(), std::back_inserter(output),
					   [](std::string &sr)
					   { return std::stof(sr); });
	else if (typeid(T) == typeid(double))
		std::transform(str_output.begin(), str_output.end(), std::back_inserter(output),
					   [](std::string &sr)
					   { return std::stod(sr); });
	else
		std::cout << "Error: unsupportted return template type" << std::endl;

	return output;
}