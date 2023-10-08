from pyworkforce.queuing import ErlangC



# Transactions: Number of incoming requests
# Resource: The element that handles a transaction
# Arrival rate: The number of incoming transactions in a time interval
# Average speed of answer (ASA): Average time that a transaction waits in the queue to be attended by a resource
# Average handle time (AHT): Average time that takes to a single resource to attend a transaction
#
# Other variables are in the diagram, but that is important for the model, those are:
# 
# Shrinkage: Expected percentage of time that a server is not available, for example, due to breaks, scheduled training, etc.
# Occupancy: Percentage of time that a resource is handling a transaction
# Service level: Percentage of transactions that arrives at a resource before a target ASA



erlang = ErlangC(transactions=10, asa=20/60, aht=3, interval=30, shrinkage=0)

print("service level",erlang.service_level(positions=10))
print("waiting probability",erlang.waiting_probability(positions=10))
