#ifndef MPI_COMM_H
#define MPI_COMM_H

#include "common_config.h"

/*
 * the following pragma prevents preprocessor to report warning about
 * unused parameters; especially from OpenMPI header comm_inln.h
 *
 * Note that nvcc does seem to recognize this pragma and it appears to
 * be a bug; see http://forums.nvidia.com/index.php?showtopic=163810
 */
#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif // __GNUC__
#include <mpi.h>

namespace mpiUtils
{
  /**
   * \brief Object representation of an MPI communicator.
   *
   * At present, groups are not implemented so the only communicator
   * is MPI_COMM_WORLD.
   */
  class MpiComm
  {
  public:
    //! Empty constructor builds an object for MPI_COMM_WORLD
    MpiComm();

    //! Construct a MpiComm for a given MPI communicator
    MpiComm(MPI_Comm comm);

    //! Empty destructor
    virtual ~MpiComm();

    //! Get an object representing MPI_COMM_WORLD
    static MpiComm &world();

    //! Get an object representing MPI_COMM_SELF
    static MpiComm &self();

    //! Return process rank
    int getRank() const { return myRank_; }

    //! Return number of processors in the communicator
    int getNProc() const { return nProc_; }

    //! Synchronize all the processors in the communicator
    void synchronize() const;

    //! @name Collective communications
    //@{

    //! All-to-all gather-scatter
    void allToAll(void *sendBuf, int sendCount, int sendType,
                  void *recvBuf, int recvCount, int recvType) const;

    //! Variable-length gather-scatter
    void allToAllv(void *sendBuf, int *sendCount, int *sendDisplacements, int sendType,
                   void *recvBuf, int *recvCount, int *recvDisplacements, int recvType) const;

    //! Do a collective operation, scattering the results to all processors
    void allReduce(void *input, void *result, int inputCount, int type, int op) const;

    //! Gather to root
    void gather(void *sendBuf, int sendCount, int sendType,
                void *recvBuf, int recvCount, int recvType, int root) const;

    //! Gather variable-sized arrays to root
    void gatherv(void *sendBuf, int sendCount, int sendType,
                 void *recvBuf, int *recvCount, int *displacements, int recvType, int root) const;

    //! Gather to all processors
    void allGather(void *sendBuf, int sendCount, int sendType,
                   void *recvBuf, int recvCount, int recvType) const;

    //! Variable-length gather to all processors
    void allGatherv(void *sendBuf, int sendCount, int sendType,
                    void *recvBuf, int *recvCount, int *recvDisplacements, int recvType) const;

    //! Broadcast
    void bcast(void *msg, int length, int type, int src) const;
    //@}

    //! @name Point-to-point communications
    //@{
    void send(void *sendBuf, int sendCount, int sendType, int dest, int tag) const;

    void recv(void *recvBuf, int recvCount, int recvtype, int source, int tag) const;

    void sendrecv(void *sendBuf, int sendCount, int sendType, int dest, int sendtag,
                  void *recvBuf, int recvCount, int recvType, int src, int recvtag) const;

    MPI_Request Isend(void *sendBuf, int sendCount, int sendType, int dest, int tag) const;

    MPI_Request Ibsend(void *sendBuf, int sendCount, int sendType, int dest, int tag) const;

    MPI_Request Issend(void *sendBuf, int sendCount, int sendType, int dest, int tag) const;

    MPI_Request Irsend(void *sendBuf, int sendCount, int sendType, int dest, int tag) const;

    MPI_Request Irecv(void *recvBuf, int recvCount, int recvType, int source, int tag) const;
    //@}

    //! Get the MPI_Comm communicator handle
    MPI_Comm getComm() const { return comm_; }

    //! @name Data types
    //@{
    //! Integer data type
    static const int INT;
    //! Float data type
    static const int FLOAT;
    //! Double data type
    static const int DOUBLE;
    //! Character data type
    static const int CHAR;
    //@}

    //! @name Operations
    //@{
    //! Summation operation
    static const int SUM;
    //! Minimize operation
    static const int MIN;
    //! Maximize operation
    static const int MAX;
    //! Dot-product (Multiplication) operation
    static const int PROD;
    //@}

    // errCheck() checks the return value of an MPI call and throws
    // a ParallelException upon failure.
    static void errCheck(int errCode, const std::string &methodName);

    //! Converts a PMachine data type code to a MPI_Datatype
    static MPI_Datatype getDataType(int type);

    //! Converts a PMachine operator code to a MPI_Op operator code.
    static MPI_Op getOp(int op);

  protected:
    MPI_Comm comm_;

    int nProc_;
    int myRank_;
    /** common initialization function, called by all ctors */
    void init();
    /** Indicate whether MPI is currently running */
    int mpiIsRunning() const;
  };
}
#endif
