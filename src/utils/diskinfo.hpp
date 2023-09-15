#include <string>
#include <limits.h>
#include <unistd.h>
#include <iostream>
#include <sys/statfs.h>

namespace disk
{
	enum memType
	{
		B = 1,
		KB = 10,
		MB = 20,
		GB = 30
	};
};

class DiskInfo
{
private:
	struct statfs buf;

public:
	std::string path;

	DiskInfo() : path("./"){};
	DiskInfo(const std::string &path_) : path(path_){};

	template <disk::memType T>
	bool avail(unsigned long long consume)
	{
		int out = statfs(path.c_str(), &buf);
		unsigned long long blocksize = buf.f_bsize;					 // 每个block里包含的字节数
		unsigned long long totalsize = blocksize * buf.f_blocks;	 // 总的字节数，f_blocks为block的数目
		unsigned long long freeDisk = buf.f_bfree * blocksize;		 // 剩余空间的大小
		unsigned long long availableDisk = buf.f_bavail * blocksize; // 可用空间大小

		if (out != -1)
		{
			// unsigned long long avK = availableDisk >> 10;
			// unsigned long long avM = availableDisk >> 20;
			// unsigned long long avG = availableDisk >> 30;
			unsigned long long availDisk = availableDisk >> T;
			bool temp = availDisk > consume;
			return temp;
		}
		else
		{
			std::cerr << "Error: statfs failed for path " << path << std::endl;
			return false;
		}
	};
};

template <disk::memType T>
bool disk_avail(const std::string &path_, const long long need, std::string tag)
{
	DiskInfo disk(path_);

	while (!disk.avail<T>(need))
	{
		std::cout << "Note: Waiting disk space for " << tag << std::endl;
		sleep(600);
	}

	return true;
}
