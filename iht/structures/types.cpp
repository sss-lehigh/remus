#include "hashtable.h"
#include "iht_ds.h"
#include "linked_set.h"
#include "common.h"

template class RdmaIHT<int, int, CNF_ELIST_SIZE, CNF_PLIST_SIZE>;
template class Hashtable<int, int, CNF_PLIST_SIZE>;
template class LinkedSet<int, int>;