#ifndef ERROR_POLLING_H
#define ERROR_POLLING_H

#include "common_config.h"
#include "TestForException.h"
// #include "MpiComm.h"

namespace mpiUtils
{
  class MpiComm;

  /** \brief ErrorPolling provides utilities for establishing agreement
   * between processors on whether an std::exception has been detected on any one
   * processor.
   *
   * The two functions must be used in a coordinated way. The simplest use
   * case is to embed a call to reportFailure() whenever an std::exception is
   * detected at the top-level try/catch block, and then to do a call to
   * pollForFailures() whenever it is desired to check for off-processor
   * errors before proceeding. The macro

    \code
    TEST_FOR_FAILURE(comm);
    \endcode

   * calls pollForFailures() and throws an std::exception if the return value is
   * true.
   *
   * Polling is a collective operation (an MPI_Reduce) and so incurs some
   * performance overhead. It can be disabled with a call to
   * \code
   * mpiUtils::ErrorPolling::disable();
   * \endcode
   * IMPORTANT: all processors must agree on whether collective error checking
   * is enabled or disabled. If there are inconsistent states, the reduction
   * operations in pollForFailures() will hang because some processors cannot be
   * contacted.
   */
  class ErrorPolling
  {
  public:
    /** Call this function upon catching an std::exception in order to
     * inform other processors of the error. This function will do an
     * AllReduce in conjunction with calls to either this function or
     * its partner, pollForFailures(), on the other processors. This
     * procedure has the effect of communicating to the other
     * processors that an std::exception has been detected on this one. */
    static void reportFailure(const MpiComm &comm);

    /** Call this function after std::exception-free completion of a
     * try/catch block. This function will do an AllReduce in
     * conjunction with calls to either this function or its partner,
     * reportFailure(), on the other processors. If a failure has been
     * reported by another processor, the call to pollForFailures()
     * will return true and an std::exception can be thrown. */
    static bool pollForFailures(const MpiComm &comm);

    /** Activate error polling */
    static void enable() { isActive() = true; }

    /** Disable error polling */
    static void disable() { isActive() = false; }

  private:
    /** Set or check whether error polling is active */
    static bool &isActive()
    {
      static bool rtn = true;
      return rtn;
    }
  };

  /**
   * This macro polls all processors in the given communicator to find
   * out whether an error has been reported by a call to
   * ErrorPolling::reportFailure(comm).
   *
   * @param comm [in] The communicator on which polling will be done
   */
#define POLL_FOR_FAILURES(comm)                                     \
  TEST_FOR_EXCEPTION(mpiUtils::ErrorPolling::pollForFailures(comm), \
                     std::runtime_error,                            \
                     "off-processor error detected by proc=" << (comm).getRank());
}

#endif // ERROR_POLLING_H
