#include "ErrorPolling.h"
#include "MpiComm.h"

namespace mpiUtils
{
  void ErrorPolling::reportFailure(const MpiComm &comm)
  {
    if (isActive())
    {
      int myBad = 1;
      int anyBad = 0;
      comm.allReduce((void *)&myBad, (void *)&anyBad, 1, MpiComm::INT,
                     MpiComm::SUM);
    }
  }

  bool ErrorPolling::pollForFailures(const MpiComm &comm)
  {
    /* bypass if inactive */
    if (!isActive())
      return true;

    int myBad = 0;
    int anyBad = 0;
    try
    {
      comm.allReduce((void *)&myBad, (void *)&anyBad, 1, MpiComm::INT,
                     MpiComm::SUM);
    }
    catch (const std::exception &)
    {
      return true;
    }
    return anyBad > 0;
  }
} // namespace mpiUtils
