
typedef unsigned InTy;
typedef unsigned OutTy;
typedef unsigned size_type;

void prefix_sum(size_type length, const InTy* in, OutTy *prefix) {
  OutTy total = 0;
  for (size_type n = 0; n < length; n++) {
    prefix[n] = total;
    total += (OutTy)in[n];
  }
  prefix[length] = total;
}

