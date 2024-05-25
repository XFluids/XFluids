#include "kinetics1.hpp"
#include "cantera_interface.h"

using namespace Cantera;

CanteraInterface::CanteraInterface()
{
	int retn = kinetics1(0, 0);
	appdelete();
}

CanteraInterface::~CanteraInterface()
{
}