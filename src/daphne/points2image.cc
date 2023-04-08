#include <cmath>
#include <iostream>
#include <cstring>
#include "points2image.h"

// Reads the next point cloud
void parsePointCloud(std::ifstream& input_file, PointCloud2* pointcloud2) {
  input_file.read((char*)&(pointcloud2->height), sizeof(int));
  input_file.read((char*)&(pointcloud2->width), sizeof(int));
  input_file.read((char*)&(pointcloud2->point_step), sizeof(uint));
#ifdef DEBUG
  printf("PointCloud: height=%d width=%d point_step=%d\n",
          pointcloud2->height , pointcloud2->width , pointcloud2->point_step);
#endif
  pointcloud2->data = (float*) malloc(pointcloud2->height * pointcloud2->width * pointcloud2->point_step * sizeof(float));
  input_file.read((char*)pointcloud2->data, pointcloud2->height * pointcloud2->width * pointcloud2->point_step);
}

// Parses the next camera extrinsic matrix.
void  parseCameraExtrinsicMat(std::ifstream& input_file, Mat44* cameraExtrinsicMat) {
  try {
    for (int h = 0; h < 4; h++)
      for (int w = 0; w < 4; w++)
        input_file.read((char*)&(cameraExtrinsicMat->data[h][w]),sizeof(double));
  } catch (std::ifstream::failure) {
    throw std::ios_base::failure("Error reading the next extrinsic matrix.");    
  }
}

// Parses the next camera matrix.
void parseCameraMat(std::ifstream& input_file, Mat33* cameraMat ) {
  try {
    for (int h = 0; h < 3; h++)
      for (int w = 0; w < 3; w++)
        input_file.read((char*)&(cameraMat->data[h][w]), sizeof(double));
  } catch (std::ifstream::failure) {
    throw std::ios_base::failure("Error reading the next camera matrix.");
  }
}

// Parses the next distance coefficients.
void  parseDistCoeff(std::ifstream& input_file, Vec5* distCoeff) {
  try {
    for (int w = 0; w < 5; w++)
      input_file.read((char*)&(distCoeff->data[w]), sizeof(double));
  } catch (std::ifstream::failure) {
    throw std::ios_base::failure("Error reading the next set of distance coefficients.");
  }
}

// Parses the next image sizes.
void  parseImageSize(std::ifstream& input_file, ImageSize* imageSize) {
  try {
    input_file.read((char*)&(imageSize->width), sizeof(int));
    input_file.read((char*)&(imageSize->height), sizeof(int));
  } catch (std::ifstream::failure) {
    throw std::ios_base::failure("Error reading the next image size.");
  }
}

// Parses the next reference image.
void parsePointsImage(std::ifstream& output_file, PointsImage* goldenResult) {
  try {
    // read data of static size
    output_file.read((char*)&(goldenResult->image_width), sizeof(int));
    output_file.read((char*)&(goldenResult->image_height), sizeof(int));
    output_file.read((char*)&(goldenResult->max_y), sizeof(int));
    output_file.read((char*)&(goldenResult->min_y), sizeof(int));
    int pos = 0;
    int elements = goldenResult->image_height * goldenResult->image_width;
    goldenResult->intensity = new float[elements];
    goldenResult->distance = new float[elements];
    goldenResult->min_height = new float[elements];
    goldenResult->max_height = new float[elements];
    // read data of variable size
    for (int h = 0; h < goldenResult->image_height; h++) {
      for (int w = 0; w < goldenResult->image_width; w++) {
        output_file.read((char*)&(goldenResult->intensity[pos]), sizeof(float));
        output_file.read((char*)&(goldenResult->distance[pos]), sizeof(float));
        output_file.read((char*)&(goldenResult->min_height[pos]), sizeof(float));
        output_file.read((char*)&(goldenResult->max_height[pos]), sizeof(float));
        pos++;
      }
    }
  } catch (std::ios_base::failure) {
    throw std::ios_base::failure("Error reading the next reference image.");
  }
}

// return how many could be read
int points2image::read_next_testcases(int count) {
  // free the memory that has been allocated in the previous iteration
  // and allocate new for the currently required data sizes
  //if (pointcloud2) 
  //  for (int m = 0; m < count; ++m)
  //    delete [] pointcloud2[m].data;
  delete [] pointcloud2;
  pointcloud2 = new PointCloud2[count];
  delete [] cameraExtrinsicMat;
  cameraExtrinsicMat = new Mat44[count];
  delete [] cameraMat;
  cameraMat = new Mat33[count];
  delete [] distCoeff;
  distCoeff = new Vec5[count];
  delete [] imageSize;
  imageSize = new ImageSize[count];
  delete [] results;
  results = new PointsImage[count];

  // read data from the next test case
  int i;
  for (i = 0; (i < count) && (read_testcases < testcases); i++,read_testcases++) {
    try {
      parsePointCloud(input_file, pointcloud2 + i);
      parseCameraExtrinsicMat(input_file, cameraExtrinsicMat + i);
      parseCameraMat(input_file, cameraMat + i);
      parseDistCoeff(input_file, distCoeff + i);
      parseImageSize(input_file, imageSize + i);
    } catch (std::ios_base::failure& e) {
      std::cerr << e.what() << std::endl;
      exit(-3);
    }
  }
  return i;
}

int points2image::read_number_testcases(std::ifstream& input_file) {
  // reads the number of testcases in the data stream
  int number = 0;
  try {
    input_file.read((char*)&(number), sizeof(int));
  } catch (std::ifstream::failure) {
    throw std::ios_base::failure("Error reading the number of testcases.");
  }
  return number;
}

void points2image::init() {
  std::cout << "Open testcase and reference data streams\n";
  input_file.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
  output_file.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
  try {
    input_file.open("./p2i_input.dat", std::ios::binary);
  } catch (std::ifstream::failure) {
    std::cerr << "Error opening the input data file" << std::endl;
    exit(-2);
  }
  try {
    output_file.open("./p2i_output.dat", std::ios::binary);
  } catch (std::ifstream::failure) {
    std::cerr << "Error opening the output data file" << std::endl;
    exit(-2);
  }
  try {
    // consume the total number of testcases
    testcases = read_number_testcases(input_file);
    printf("the total number of testcases = %d\n", testcases);
  } catch (std::ios_base::failure& e) {
    std::cerr << e.what() << std::endl;
    exit(-3);
  }

  // prepare the first iteration
  error_so_far = false;
  max_delta = 0.0;
  pointcloud2 = NULL;
  cameraExtrinsicMat = NULL;
  cameraMat = NULL;
  distCoeff = NULL;
  imageSize = NULL;
  results = NULL;

  std::cout << "Done\n" << std::endl;
}

/**
 * This code is extracted from Autoware, file:
 * ~/Autoware/ros/src/sensing/fusion/packages/points2image/lib/points_image/points_image.cpp
 * It uses the test data that has been read before and applies the linked algorithm.
 * pointcloud2: cloud of points to transform
 * cameraExtrinsicMat: camera matrix used for transformation
 * cameraMat: camera matrix used for transformation
 * distCoeff: distance coefficients for cloud transformation
 * imageSize: the size of the resulting image
 * returns: the two dimensional image of transformed points
 */
PointsImage pointcloud2_to_image(
    const PointCloud2& pointcloud2,
    const Mat44& cameraExtrinsicMat,
    const Mat33& cameraMat, const Vec5& distCoeff,
    const ImageSize& imageSize) {
  // initialize the resulting image data structure
  int w = imageSize.width;
  int h = imageSize.height;
  PointsImage msg;
  msg.max_y = -1;
  msg.min_y = h;
  msg.image_height = imageSize.height;
  msg.image_width = imageSize.width;
  msg.intensity = new float[w*h];
  std::memset(msg.intensity, 0, sizeof(float)*w*h);
  msg.distance = new float[w*h];
  std::memset(msg.distance, 0, sizeof(float)*w*h);
  msg.min_height = new float[w*h];
  std::memset(msg.min_height, 0, sizeof(float)*w*h);
  msg.max_height = new float[w*h];
  std::memset(msg.max_height, 0, sizeof(float)*w*h);
  int32_t max_y = -1;
  int32_t min_y = h;

  // preprocess the given matrices
  Mat33 invR;
  Mat13 invT;
  // transposed 3x3 camera extrinsic matrix
  for (int row = 0; row < 3; row++)
    for (int col = 0; col < 3; col++)
      invR.data[row][col] = cameraExtrinsicMat.data[col][row];
  // translation vector: (transposed camera extrinsic matrix)*(fourth column of camera extrinsic matrix)
  for (int row = 0; row < 3; row++) {
    invT.data[row] = 0.0;
    for (int col = 0; col < 3; col++)
      invT.data[row] -= invR.data[row][col] * cameraExtrinsicMat.data[col][3];
  }

  // call the kernel
  compute_point_from_pointcloud(
      pointcloud2.data, msg.distance, msg.intensity, msg.min_height, msg.max_height,
      pointcloud2.width, pointcloud2.height, pointcloud2.point_step, w, h,
      invR, invT, distCoeff, cameraMat, &min_y, &max_y);

  msg.max_y = max_y;
  msg.min_y = min_y;
  return msg;
}

void points2image::run(int p) {
  while (read_testcases < testcases) {
    int count = read_next_testcases(p);
    // run the algorithm for each input data set
    for (int i = 0; i < count; i++) {
      results[i] = pointcloud2_to_image(
          pointcloud2[i], cameraExtrinsicMat[i],
          cameraMat[i], distCoeff[i], imageSize[i]);
    }
    // compare with the reference data
    check_next_outputs(count);
  }
}

void points2image::check_next_outputs(int count) {
  PointsImage reference;
  // parse the next reference image
  // and compare it to the data generated by the algorithm
  for (int i = 0; i < count; i++) {
    try {
      parsePointsImage(output_file, &reference);
    } catch (std::ios_base::failure& e) {
      std::cerr << e.what() << std::endl;
      exit(-3);
    }
    // detect image size deviation
    if ((results[i].image_height != reference.image_height)
        || (results[i].image_width != reference.image_width)) {
      error_so_far = true;
    }
    // detect image extend deviation
    if ((results[i].min_y != reference.min_y)
        || (results[i].max_y != reference.max_y)) {
      error_so_far = true;
    }
    // compare all pixels
    int pos = 0;
    for (int h = 0; h < reference.image_height; h++)
      for (int w = 0; w < reference.image_width; w++) {
        // compare members individually and detect deviations
        if (std::fabs(reference.intensity[pos] - results[i].intensity[pos]) > max_delta)
          max_delta = fabs(reference.intensity[pos] - results[i].intensity[pos]);
        if (std::fabs(reference.distance[pos] - results[i].distance[pos]) > max_delta)
          max_delta = fabs(reference.distance[pos] - results[i].distance[pos]);
        if (std::fabs(reference.min_height[pos] - results[i].min_height[pos]) > max_delta)
          max_delta = fabs(reference.min_height[pos] - results[i].min_height[pos]);
        if (std::fabs(reference.max_height[pos] - results[i].max_height[pos]) > max_delta)
          max_delta = fabs(reference.max_height[pos] - results[i].max_height[pos]);
        pos++;
      }
    // free the memory allocated by the reference image read above
    delete [] reference.intensity;
    delete [] reference.distance;
    delete [] reference.min_height;
    delete [] reference.max_height;
  }
}

bool points2image::check_output() {
  std::cout << "checking output \n";
  input_file.close();
  output_file.close();
  std::cout << "max delta: " << max_delta << "\n";
  if ((max_delta > MAX_EPS) || error_so_far) {
    return false;
  } else {
    return true;
  }
}

