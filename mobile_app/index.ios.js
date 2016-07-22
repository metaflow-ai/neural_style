import React, { Component } from 'react'
import ReactNative from 'react-native'

import {
  AppRegistry,
  StyleSheet,
  CameraRoll,
  Image,
  Text,
  View,
  NativeModules,
  TextInput,
  Dimensions
} from 'react-native'

import Camera from 'react-native-camera'


class ReactNativeTF extends Component {

  constructor(props) {
    super(props)

    this.state = {
      v1: 0.,
      v2: 0.,
      result: 0.,
      camera: null,
      imagePath: '',
      image: null,
    }

    this.takePicture = this.takePicture.bind(this)
    this.resetPicture = this.resetPicture.bind(this)
  }


  takePicture() {
    this.state.camera.capture()
      .then((data) => {
        console.log("taking picture ... ", data.path)

        const IM = NativeModules.ImageManager
        IM.chill(data.path, (err, data) => {
          this.setState({
            image: "data:image/png;base64," + data
          })
        })
      })
      .catch(err => console.error(err));
  }


  resetPicture() {
    this.setState({
      image: null
    });
  }


  render() {
    return (
      <View style={styles.container}>
        <View>
          <Text style={styles.reset} onPress={this.resetPicture}>[RESET]</Text>
        </View>
        {this.state.image === null ? (
          <Camera
            ref={(cam) => {
              this.state.camera = cam;
            }}
            style={styles.preview}
            aspect={Camera.constants.Aspect.fill}
            captureTarget={Camera.constants.CaptureTarget.tmp}
            captureQuality="high"
          >
            <Text style={styles.capture} onPress={this.takePicture.bind(this)}>[CAPTURE]</Text>
          </Camera>
        ) : (
          <View>
            <Image
              source={{uri: this.state.image}}
              style={styles.cropped}
            />
          </View>
        )}
      </View>
    );
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  preview: {
    flex: 1,
    justifyContent: 'flex-end',
    alignItems: 'center',
    height: Dimensions.get('window').height,
    width: Dimensions.get('window').width
  },
  cropped: {
    alignSelf: 'center',
    width: 300,
    height: 300,
    borderColor: 'red',
    borderWidth: 3
  },
  capture: {
    flex: 0,
    backgroundColor: '#fff',
    borderRadius: 5,
    color: '#000',
    padding: 10,
    margin: 40
  },
  reset: {
    flex: 0,
    backgroundColor: '#fff',
    borderRadius: 5,
    color: '#000',
    padding: 10,
    margin: 20
  }
});

AppRegistry.registerComponent('ReactNativeTF', () => ReactNativeTF);
