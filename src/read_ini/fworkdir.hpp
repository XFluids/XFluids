#pragma once

#include <unistd.h>

#include "global_setup.h"

#if USE_CXX_BOOST
#include <boost/filesystem.hpp>
namespace fsm = boost::filesystem;
#else
#include <filesystem>
namespace fsm = std::filesystem;
#endif // end USE_CXX_BOOST

// =======================================================
// // // set work dir
// =======================================================
std::string getWorkDir(std::string exe_path, std::string exe_name)
{
	char cwd[1000];
	getcwd(cwd, 1000);
	exe_path = std::string(cwd) + "/" + exe_path;
	do
	{
		int a = exe_path.find_last_of("/");
		// std::cout << exe_path.erase(a) + "/runtime.dat" << std::endl;
		if (fsm::exists(exe_path.erase(a) + "/runtime.dat"))
			return exe_path;
		if (a < 0)
			break;
	} while (1);
	std::cout << "Error: cannot find WorkDir, run executable file under Program Directory." << std::endl;
	exit(EXIT_FAILURE);
	return "Error!!!";
}