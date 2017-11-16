#include <vector>

//#include "caffe/filler.hpp"
#include "caffe/layers/dense_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Compute_distance_data_gpu(int nthreads, const int K, const Dtype* bottom,
                                          const Dtype* label,  Dtype* distance, const float dense_weight) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    distance[index] = bottom[index] - label[index] * dense_weight;
  }
}

template <typename Dtype>
__global__ void Compute_center_diff_gpu(int nthreads, const int M, const int K, 
        const Dtype* label, const Dtype* distance, Dtype* variation_sum, 
        Dtype* center_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int count = 0;
    for (int m = 0; m < M; m++) {
      const int label_value = static_cast<int>(label[m]);
      if (label_value == index) {
        count++;
        for (int k = 0; k < K; k++) {
          variation_sum[index * K + k] -= distance[m * K + k];
        }
      }
    }
    for (int k = 0; k < K; k++) {
      center_diff[index * K + k] = variation_sum[index * K + k] /(count + (Dtype)1.);
    }
  }
}


template <typename Dtype>
void DenseLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int nthreads = M_ * K_;
  const float dense_weight = this->layer_param_.dense_loss_param().dense_weight();
  
  Compute_distance_data_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
                                distance_.mutable_gpu_data(),dense_weight);
  Dtype dot;
  caffe_gpu_dot(M_ * K_, distance_.gpu_data(), distance_.gpu_data(), &dot);
  Dtype loss = dot / M_ / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void DenseLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  /*int nthreads = N_;
  caffe_gpu_set(N_ * K_, (Dtype)0., variation_sum_.mutable_cpu_data());
  Compute_center_diff_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, M_, K_, bottom[1]->gpu_data(), distance_.gpu_data(), 
                                variation_sum_.mutable_cpu_data(), this->blobs_[0]->mutable_gpu_diff());
  */
  if (propagate_down[0]) 
  {
    caffe_gpu_scale(M_ * K_, top[0]->cpu_diff()[0] / M_, 
                             distance_.gpu_data(), bottom[0]->mutable_gpu_diff());
  }
  if (propagate_down[1]) 
  {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DenseLossLayer);

}  // namespace caffe


