#include "common.h"
#include "hashtable.h"
#include "iht_ds.h"
#include "linked_set.h"

// [mfs] I'm not sure why this is needed

template class RdmaIHT<int, int, CNF_ELIST_SIZE, CNF_PLIST_SIZE>;
template class Hashtable<int, int, CNF_PLIST_SIZE>;
template class LinkedSet<int, int>;