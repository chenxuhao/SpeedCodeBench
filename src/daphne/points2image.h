//#include "benchmark.h"
#include <fstream>
#include <string.h>
#include "datatypes.h"

// maximum allowed deviation from the reference results
#define MAX_EPS 0.001
// number of GPU threads
#define THREADS 256

void compute_point_from_pointcloud(
    const float*  __restrict cp, 
          float* volatile msg_distance,
          float* volatile msg_intensity,
          float* __restrict msg_min_height,
          float* __restrict msg_max_height,
    int width, int height, int point_step,
    int w, int h, Mat33 invR, Mat13 invT,
    Vec5 distCoeff, Mat33 cameraMat,
    int* __restrict min_y, int* __restrict max_y);
 
class points2image {
  private:
    // the number of testcases read
    int read_testcases = 0;
    // testcase and reference data streams
    std::ifstream input_file, output_file;
    // whether critical deviation from the reference data has been detected
    bool error_so_far = false;
    // deviation from the reference data
    double max_delta = 0.0;
  public:
    int testcases = 1;
    /*
     * Initializes the kernel. Must be called before run().
     */
    void init();
    /**
     * Performs the kernel operations on all input and output data.
     * p: number of testcases to process in one step
     */
    void run(int p = 1);
    /**
     * Finally checks whether all input data has been processed successfully.
     */
    bool check_output();
    // the point clouds to process in one iteration
    PointCloud2* pointcloud2 = NULL;
    // the associated camera extrinsic matrices
    Mat44* cameraExtrinsicMat = NULL;
    // the associated camera intrinsic matrices
    Mat33* cameraMat = NULL;
    // distance coefficients for the current iteration
    Vec5* distCoeff = NULL;
    // image sizes for the current iteration
    ImageSize* imageSize = NULL;
    // Algorithm results for the current iteration
    PointsImage* results = NULL;
  protected:
    /**
     * Reads the next test cases.
     * count: the number of testcases to read
     * returns: the number of testcases actually read
     */
    int read_next_testcases(int count);
    /**
     * Compares the results from the algorithm with the reference data.
     * count: the number of testcases processed 
     */
    void check_next_outputs(int count);
    /**
     * Reads the number of testcases in the data set.
     */
    int  read_number_testcases(std::ifstream& input_file);
};

