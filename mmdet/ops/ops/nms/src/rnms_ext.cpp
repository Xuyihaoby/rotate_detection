// Modified from https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/cpu_nms.pyx, Soft-NMS is added
// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <torch/extension.h>

torch::Tensor rnms_cpu(const torch::Tensor &dets_tsr, const float threshold);

#ifdef WITH_CUDA
at::Tensor rnms_cuda(const at::Tensor& dets, const float threshold);
#endif

at::Tensor rnms(const at::Tensor& dets, const float threshold){
  if (dets.device().is_cuda()) {
#ifdef WITH_CUDA
    return rnms_cuda(dets, threshold);
#else
    AT_ERROR("nms is not compiled with GPU support");
#endif
  }
  return rnms_cpu(dets, threshold);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rnms", &rnms, "non-maximum suppression for rotated bounding boxes");
}
