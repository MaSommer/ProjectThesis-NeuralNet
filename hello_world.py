"""
The following program distributes the calculation of the average of the numbers in a list
It requires MPI to be installed as well as mpi4py. mpi4py can be installed using pip.
Installing MPI is a bit more complicated and can be done by following these steps:
    1. Download the software from https://www.open-mpi.org/software/ompi/v2.0/
    2. Unzip the downloaded folder
    3. Open terminal and navigate to the unzipped folder
    4. Type into the terminal: make all
    5. Type into the terminal: make install
    6. Test that everything went well by typing: $HOME/opt/usr/local/bin/mpirun --version

The program is run by calling:
    mpiexec -n 4 python hello_world.py
4 indicates that 4 processors are being used. 
"""
import TestObject as to

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()     # Total number of processors
rank = comm.Get_rank()     # Unique ID of current processor

data_range = 1500           # Numbers to average

def calc_average(a):
    return sum(a) / float(len(a))

# Work done by the master processor
if rank == 0:
    data = [i for i in range(data_range+1)]    # Read data from file
    data_sizes = int(data_range / size)        # Calculated how much of the data each processor is to process

    # Distributes the data to the other processors
    for i in range(1, size):
        #comm.send(data[i*data_sizes:(i+1)*data_sizes+1], dest=i, tag=11)
        comm.send(to.TestObject(data[i*data_sizes:(i+1)*data_sizes+1]), dest=i, tag=11)

    # Calculates the average
    average = calc_average(to.TestObject(data[:data_sizes+1]).get_list())


# Work done by slave processors
else:
    my_data = comm.recv(source=0, tag=11)   # Recieve data from master
    average = calc_average(my_data.get_list())         # Calculate result
    comm.send(average, dest=0, tag=11)      # Send result to master


# Master prints its own and the other results
if rank == 0:
    sum_of_averages = average
    for i in range(1, size):
        status = MPI.Status()
        recv_data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        print("Got data: " + str(recv_data) + ", from processor: " + str(status.Get_source()))
        sum_of_averages += recv_data
    
    average = sum_of_averages / float(size)
    print("\n### RESULT ###")
    print("Average: " + str(average))
    
