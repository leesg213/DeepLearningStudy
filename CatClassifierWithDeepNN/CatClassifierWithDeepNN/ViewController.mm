//
//  ViewController.m
//  CatClassifierWithDeepNN
//
//  Created by Sugu Lee on 6/21/26.
//

#import "ViewController.h"
#include <vector>
#include <H5Cpp.h>
#include "OneHiddenLayerModel.hpp"

@implementation ViewController
{
    IBOutlet NSImageView* imageView;
    
    size_t num_trainset, num_testset;
    size_t num_features;
    std::vector<float> trainset_x, testset_x;
    std::vector<uint8_t> trainset_y, testset_y;
}

- (void)viewDidLoad {
    [super viewDidLoad];

    // Do any additional setup after loading the view.
    [self loadDataSets];
}


- (void)setRepresentedObject:(id)representedObject {
    [super setRepresentedObject:representedObject];

    // Update the view, if already loaded.
}

- (void)loadDataSets
{
    NSString* dataSetPath_train = @"/Users/sugulee/Documents/Files/home/jovyan/work/release/W2A2/datasets/train_catvnoncat.h5";
    NSString* dataSetPath_test = @"/Users/sugulee/Documents/Files/home/jovyan/work/release/W2A2/datasets/test_catvnoncat.h5";

    H5::H5File train_file(dataSetPath_train.UTF8String, H5F_ACC_RDONLY);
    H5::H5File test_file(dataSetPath_test.UTF8String, H5F_ACC_RDONLY);

    H5::DataSet ds_train_set_x = train_file.openDataSet("train_set_x"); // 209,64,64,3
    H5::DataSet ds_train_set_y = train_file.openDataSet("train_set_y"); // 209

    H5::DataSet ds_test_set_x = test_file.openDataSet("test_set_x");
    H5::DataSet ds_test_set_y = test_file.openDataSet("test_set_y");
    
    H5::DataSpace dataspace_train = ds_train_set_x.getSpace();
    H5::DataSpace dataspace_test = ds_test_set_x.getSpace();

    hsize_t dims[4];
    dataspace_train.getSimpleExtentDims(dims, nullptr);

    int numImages_train = dims[0];
    int height    = dims[1];
    int width     = dims[2];
    int channels  = dims[3];

    dataspace_test.getSimpleExtentDims(dims, nullptr);
    int numImages_test = dims[0];
    
    std::vector<uint8_t> train_set_x(numImages_train * width * height * channels);
    trainset_x.resize(numImages_train * width * height * channels);
    ds_train_set_x.read(train_set_x.data(), H5::PredType::NATIVE_UINT8);
    for(int i = 0;i<train_set_x.size(); ++i) trainset_x[i] = train_set_x[i]/255.0f;
    
    trainset_y.resize(numImages_train);
    ds_train_set_y.read(trainset_y.data(), H5::PredType::NATIVE_UINT8);

    std::vector<uint8_t> test_set_x(numImages_test * width * height * channels);
    testset_x.resize(numImages_test * width * height * channels);
    ds_test_set_x.read(test_set_x.data(), H5::PredType::NATIVE_UINT8);
    for(int i = 0;i<test_set_x.size(); ++i) testset_x[i] = test_set_x[i]/255.0f;
    
    testset_y.resize(numImages_test);
    ds_test_set_y.read(testset_y.data(), H5::PredType::NATIVE_UINT8);
    
    num_trainset = numImages_train;
    num_testset = numImages_test;
    num_features = width * height * channels;
    
    NSLog(@"Datasets loaded. Num train : %d Num test : %d", numImages_train, numImages_test);
}
- (IBAction)runOneHiddenLayerModel:(id)sender
{
    OneHiddenLayerModel _OneHiddenLayerModel;
    _OneHiddenLayerModel.Init();
    _OneHiddenLayerModel.TrainF(trainset_x, trainset_y, 3, num_trainset, num_features, 2500, 0.0075f);
    
    std::vector<float> results;
    _OneHiddenLayerModel.PredictF(testset_x, 3, num_testset, num_features, results);
    int num_correct = 0;
    for(int i = 0;i<results.size();++i)
    {
        int predict = results[i] >= 0.5f;
        if(predict == testset_y[i]) num_correct++;
    }
    NSLog(@"Num Correct : %d / %d", num_correct, results.size());
}
@end
