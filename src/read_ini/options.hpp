#pragma once

#include "strsplit/strsplit.h"
// ================================================================================
// // // class AppendParas Member function definitions
// // // hiding implementation of template function causes undefined linking error
// ================================================================================
class AppendParas
{
private:
	int argc;
	char **argv;

public:
	AppendParas(){};
	~AppendParas(){};
	AppendParas(int argc, char **argv) : argc(argc), argv(argv){};

	std::vector<std::string> match(std::string option)
	{
		size_t i = 0;
		option += "=";
		for (i = 0; i < argc; i++)
		{
			if (std::string(argv[i]).find(option) != std::string::npos)
			{
				return Stringsplit(std::string(argv[i]).erase(0, option.length()));
				break;
			}
		}
		return std::vector<std::string>();
	};

	template <typename T>
	std::vector<T> match(std::string option)
	{
		size_t i = 0;
		option += "=";
		for (i = 0; i < argc; i++)
		{
			if (std::string(argv[i]).find(option) != std::string::npos)
			{
				return Stringsplit<T>(std::string(argv[i]).erase(0, option.length()));
				break;
			}
		}
		return std::vector<T>();
	};

	std::vector<std::string> match(std::vector<std::string> &str, std::string option, char split = ',')
	{
		size_t i = 0;
		option += "=";
		for (i = 0; i < str.size(); i++)
		{
			if (std::string(str[i]).find(option) != std::string::npos)
			{
				return Stringsplit(std::string(str[i]).erase(0, option.length()), split);
				break;
			}
		}
		return std::vector<std::string>();
	};

	template <typename T>
	std::vector<T> match(std::vector<std::string> &str, std::string option, char split = ',')
	{
		size_t i = 0;
		option += "=";
		for (i = 0; i < str.size(); i++)
		{
			if (std::string(str[i]).find(option) != std::string::npos)
			{
				return Stringsplit<T>(std::string(str[i]).erase(0, option.length()), split);
				break;
			}
		}
		return std::vector<T>();
	};
};
