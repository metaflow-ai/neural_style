#import "RunModel.h"

#include <fstream>
#include <pthread.h>
#include <unistd.h>
#include <queue>
#include <sstream>
#include <string>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/message_lite.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

#include "ImageManager.h"

NSString* RunInferenceOnImage();


namespace {
  class IfstreamInputStream : public ::google::protobuf::io::CopyingInputStream {
  public:
    explicit IfstreamInputStream(const std::string& file_name)
    : ifs_(file_name.c_str(), std::ios::in | std::ios::binary) {}
    ~IfstreamInputStream() { ifs_.close(); }
    
    int Read(void* buffer, int size) {
      if (!ifs_) {
        return -1;
      }
      ifs_.read(static_cast<char*>(buffer), size);
      return ifs_.gcount();
    }
    
  private:
    std::ifstream ifs_;
  };
}




bool PortableReadFileToProto(const std::string& file_name,
                             ::google::protobuf::MessageLite* proto) {
  ::google::protobuf::io::CopyingInputStreamAdaptor stream(
                                                           new IfstreamInputStream(file_name));
  stream.SetOwnsCopyingStream(true);
  // TODO(jiayq): the following coded stream is for debugging purposes to allow
  // one to parse arbitrarily large messages for MessageLite. One most likely
  // doesn't want to put protobufs larger than 64MB on Android, so we should
  // eventually remove this and quit loud when a large protobuf is passed in.
  ::google::protobuf::io::CodedInputStream coded_stream(&stream);
  // Total bytes hard limit / warning limit are set to 1GB and 512MB
  // respectively.
  coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);
  return proto->ParseFromCodedStream(&coded_stream);
}



NSString* FilePathForResourceName(NSString* name, NSString* extension) {
  NSString* file_path = [[NSBundle mainBundle] pathForResource:name ofType:extension];
  if (file_path == NULL) {
    LOG(FATAL) << "Couldn't find '" << [name UTF8String] << "."
    << [extension UTF8String] << "' in bundle.";
  }
  return file_path;
}



@implementation RunModel {
}


+ (CGFloat *)transformImage:(CGFloat *)imgTensor height:(int)height width:(int)width channels:(int)channels;
{
  
  tensorflow::Tensor image_tensor(
    tensorflow::DT_FLOAT,
    tensorflow::TensorShape({1, height, width, channels})
  );

    
  auto image_tensor_mapped = image_tensor.tensor<float, 4>();
  float* out = image_tensor_mapped.data();
  
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < channels; ++c) {
        out[y*width*channels + x*channels + c] = imgTensor[y*width*channels + x*channels + c];
      }
    }
  }
  
  
  //  1. Create a Session with SessionOptions

  tensorflow::SessionOptions options; // Default Options for a Session
  tensorflow::Session* session_pointer = nullptr;
  
  tensorflow::Status session_status = tensorflow::NewSession(options, &session_pointer);
  
  if (!session_status.ok()) {
    std::string status_string = session_status.ToString();
  }
  
  std::unique_ptr<tensorflow::Session> session(session_pointer);
  LOG(INFO) << "Session created.";
  
  tensorflow::GraphDef tensorflow_graph;
  LOG(INFO) << "Graph created.";
  
  
  //  2. Load the network

  NSString* network_path = FilePathForResourceName(@"tf-frozen_overfit_8b", @"pb");
  PortableReadFileToProto([network_path UTF8String], &tensorflow_graph);
  
  LOG(INFO) << "Creating session.";
  tensorflow::Status s = session->Create(tensorflow_graph);
  
  if (!s.ok()) {
    LOG(ERROR) << "Could not create Tensorflow Graph: " << s;
  }
  
  
  //  3. Run the network
  
  std::string input_layer = "input_node:0";
  std::string output_layer = "mul_105:0";
  
  std::string keras_learning_phase = "keras_learning_phase";
  tensorflow::Tensor keras_learning_phase_tensor(
                                  tensorflow::DT_UINT8,
                                  tensorflow::TensorShape({}));
  

  std::vector<tensorflow::Tensor> outputs;
  
//  tensorflow::Status run_status = session->Run({
//    {input_layer, image_tensor}, {keras_learning_phase, keras_learning_phase_tensor}
//  }, {output_layer}, {}, &outputs);
  
  NSDate *methodStart = [NSDate date];
  
  tensorflow::Status run_status = session->Run({
    {input_layer, image_tensor}
  }, {output_layer}, {}, &outputs);
  
  NSDate *methodFinish = [NSDate date];

  NSTimeInterval executionTime = [methodFinish timeIntervalSinceDate:methodStart];
  LOG(INFO) << "executionTime: " << executionTime;

  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
  }

  tensorflow::string status_string = run_status.ToString();
  tensorflow::Tensor* output = &outputs[0];
  
  auto output_vec = output->shaped<float, 4>({1, 3, 600, 600});
  
  CGFloat * myOutput = (CGFloat *) calloc(width*height*3, sizeof(CGFloat));
  
  for (int i = 0; i < width*height*3; ++i) {
    myOutput[i] = MAX(MIN(output_vec(i), 255.), 0.);
  }
  
  //  4. Return the result
  return myOutput;
}


@end
