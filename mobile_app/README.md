# React-Native + Tensorflow

**Beware! The tensorflow master branch is moving a lot! One day you might be able to compile it, the other day you may not. We won't comment on any tensorflow compilation related issues here**

This folder is an example of how to use Tensorflow with iOS to run a neural-style transfer network.

### Tensorflow:
- Clone tensorflow: `git clone git@github.com:tensorflow/tensorflow.git && cd tensorflow` where you cloned deepback.
- Install all depedencies needed for [**compiling from sources**](https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#installing-from-sources)
- Regenerate deps files: `tensorflow/contrib/makefile/gen_file_lists.sh`
- Build the libs:
```bash
cd tensorflow/contrib/makefile

# The 'abs()' operations is missing from the build file, which is a major problem since 
# we use it, so first add this line in tf_op_files.txt
# tensorflow/core/kernels/cwise_op_abs.cc

./build_all_ios.sh
mkdir ../../../../deepback/mobile_app/lib/
cp gen/lib/libtensorflow-core.a ../../../../deepback/mobile_app/lib/
cp gen/protobuf_ios/lib/libprotobuf-lite.a ../../../../deepback/mobile_app/lib/
cp gen/protobuf_ios/lib/libprotobuf.a ../../../../deepback/mobile_app/lib/
```

If you have any trouble with the Tensorflow's iOS build, please check the official doc [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/makefile/README.md#ios)

### React-Native
Install the mobile app's dependencies
```bash
cd mobile_app && npm i
```

### Xcode
- Open the project located at `deepback/mobile_app/ios/ReactNativeTF.xcodeproj` with XCode.

If you cannot see the libRCTCamera.a added to your *Linked Frameworks and Libraries*, you should link it:

![alt text](data/Linked_Frameworks_Libraries.png "Your linked Frameworks and Libs")

```bash
npm install rnpm --global
rnpm link react-native-camera
```

or, if you'd rather [manual install](https://github.com/lwansbrough/react-native-camera#ios)
- Build the project and run it from the Simulator.

### Run the project from your phone

To run the project from your phone, you'll have to package it. Edit `deepback/mobile_app/ios/ReactNativeTF/AppDelegate.m`

Comment this line
```javascript
jsCodeLocation = [NSURL URLWithString:@"http://localhost:8081/index.ios.bundle?platform=ios&dev=true"];
```

And uncomment his line
```javascript
jsCodeLocation = [[NSBundle mainBundle] URLForResource:@"main" withExtension:@"jsbundle"];
```

Then, from XCode, change the build target to select your connected iPhone.


## Problems
- `Missing op` : Check tensorflow/contrib/makefile/tf_op_files.txt and add yout missing depepdencies