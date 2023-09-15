#include <iomanip>
#include "strsplit.h"

std::vector<std::string> Stringsplit(std::string str, const char split)
{
	std::string token;					 // recive buffers
	std::istringstream iss(str);		 // input stream
	std::vector<std::string> str_output; // recive buffers

	while (getline(iss, token, split)) // take "split" as separator
	{
		str_output.push_back(token);
	}

	return str_output;
}