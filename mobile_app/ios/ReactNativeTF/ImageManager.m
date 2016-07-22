//
//  ImageManager.m
//  ReactNativeTorch
//
//  Created by ThomasOlivier on 08/06/2016.
//  Copyright © 2016 Facebook. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "RCTLog.h"
#import "ImageManager.h"

@import AssetsLibrary;
@import UIKit;

#import "UIImage+Resize.h"
#import "RunModel.h"


@implementation ImageManager


RCT_EXPORT_MODULE();


+ (CGFloat *)getRGBAsFromImage:(UIImage*)image;
{
  int count = image.size.width * image.size.height;

  // First get the image into your data buffer
  CGImageRef imageRef = [image CGImage];
  NSUInteger width = CGImageGetWidth(imageRef);
  NSUInteger height = CGImageGetHeight(imageRef);

  CGColorSpaceRef rgb = CGColorSpaceCreateDeviceRGB();
  unsigned char * pixels = (unsigned char *) calloc(height * width * 4, sizeof(unsigned char));

  NSUInteger bytesPerPixel = 4;
  NSUInteger bytesPerRow = bytesPerPixel * width;
  NSUInteger bitsPerComponent = 8;

  CGContextRef context = CGBitmapContextCreate(pixels, width, height,
                                               bitsPerComponent, bytesPerRow, rgb,
                                               kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);

  CGContextDrawImage(context, CGRectMake(0, 0, width, height), imageRef);

  NSUInteger byteIndex = 0;
  CGFloat *aResult = (CGFloat *) calloc(height * width * 3, sizeof(CGFloat));

  for (int i = 0 ; i < count ; ++i)
  {
    aResult[i*2 + i] = (CGFloat) pixels[byteIndex + 2];
    aResult[i*2 + i+1] = (CGFloat) pixels[byteIndex + 1];
    aResult[i*2 + i+2] = (CGFloat) pixels[byteIndex];
    
    byteIndex += bytesPerPixel;
  }

  CGContextRelease(context);
  CGColorSpaceRelease(rgb);
  free(pixels);

  return aResult;
}



+ (UIImage *)getImageFromRGBA:(CGFloat*)imgTensor channels:(int)channels width:(int)width height:(int)height;
{
  UIImage* result;
  
  int newChannels = 4;
  int totalPixels = height * width * newChannels;
  CGColorSpaceRef rgb = CGColorSpaceCreateDeviceRGB();
  
  NSUInteger byteIndex = 0;
  unsigned char * pixels = (unsigned char *) calloc(totalPixels, sizeof(unsigned char));
  
  for(int i=0; i < width*height; ++i) {
    pixels[newChannels*i] = imgTensor[byteIndex + 2];
    pixels[newChannels*i+1] = imgTensor[byteIndex + 1];
    pixels[newChannels*i+2] = imgTensor[byteIndex];
    pixels[newChannels*i+3] = 1;
    
    byteIndex += channels;
  }
  
  CGDataProviderRef ref = CGDataProviderCreateWithData(NULL, pixels, width * height * newChannels, NULL);
  CGImageRef iref = CGImageCreate(width, height, 8, 8 * newChannels, width * newChannels, rgb, kCGBitmapByteOrderDefault, ref, NULL, true, kCGRenderingIntentDefault);
    
  result = [UIImage imageWithCGImage:iref];
  
  CGImageRelease(iref);
  CGDataProviderRelease(ref);
  CGColorSpaceRelease(rgb);
  
  return result;
}



RCT_EXPORT_METHOD(chill:(NSURL *)imagePath
                  callback:(RCTResponseSenderBlock)callback)
{
  ALAssetsLibrary *assetsLibrary = [[ALAssetsLibrary alloc] init];

  int channels = 3;
  int width = 600;
  int height = 600;
  
  
  [assetsLibrary assetForURL:imagePath resultBlock: ^(ALAsset *asset){
    
    ALAssetRepresentation *representation = [asset defaultRepresentation];
    CGImageRef imageRef = [representation fullResolutionImage];

    if (imageRef) {
      
      UIImageView *imageView = [[UIImageView alloc] initWithFrame:CGRectMake(0, 0, 1080, 1920)];
      imageView.image = [UIImage imageWithCGImage:imageRef scale:representation.scale orientation:representation.orientation];
      
      CGSize newSize = CGSizeMake(width, height);
      UIImage *newImage = [imageView.image resizedImage:newSize interpolationQuality:1];

      CGFloat *imgTensor = [ImageManager getRGBAsFromImage:newImage];

      CGFloat *transformedImgTensor = [RunModel transformImage:imgTensor height:height width:width channels:channels];

      UIImage *transformedImg = [ImageManager getImageFromRGBA:transformedImgTensor
                                                      channels:channels width:width height:height];

//      To save the newly created UIImage to the camera roll
//      UIImageWriteToSavedPhotosAlbum(transformedImg, nil, nil, nil);
      
      NSString *transformedImgAsBase64 = [transformedImg base64String];

      callback(@[[NSNull null], transformedImgAsBase64]);

    } else {
      RCTLogInfo(@"ERROR ELSE");
      NSLog(@"ERROR ELSE");
    }
  } failureBlock: ^(NSError *error){
    RCTLogInfo(@"FAILURE");
    NSLog(@"Failure");
  }];
}

@end