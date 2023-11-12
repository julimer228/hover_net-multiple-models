import json
import unittest
import glob
import os

class SimpleTestCase(unittest.TestCase):
    
    def testFilesExist(self):
        files = ['config_train.json', 'config_process.json', 'config_stats.json']
        for file in files:
            with self.subTest(file=file):
                self.assertTrue(os.path.isfile(file), f"JSON file '{file}' does not exist")
    
    def testTrainJSON(self):
        with open('config_train.json', 'r') as json_file:
            config_list = json.load(json_file)
        
        for config in config_list:
            with self.subTest(i=config["id"]): 
                train_dir = config["train_dir_list"][0]
                valid_dir = config["valid_dir_list"][0]
                train_data = glob.glob(train_dir+'/*.npy')
                valid_data = glob.glob(valid_dir+'/*.npy')
                id = str(config["id"])
                
                if config['dataset_name'] == "cpm17":
                    self.assertEqual(len(train_data),512,"Incorrect number of .npy files found! (train): "+id)
                    self.assertEqual(len(valid_data),512,"Incorrect number of .npy files found! (train): "+id)
                elif config['dataset_name'] == "kumar":
                    self.assertEqual(len(train_data),784,"Incorrect number of .npy files found! (train): "+id)
                    self.assertEqual(len(valid_data),686,"Incorrect number of .npy files found! (train): "+id)
                else:
                    self.fail("Incorrect dataset name!")
    
    def testProcessJSON(self):
        with open('config_process.json', 'r') as json_file:
            config_list = json.load(json_file)
        
        for config in config_list:
            with self.subTest(i=config["id"]): 
                
                input_dir_test = config["input_dir_test"]
                input_dir_train = config["input_dir_train"]
                id = str(config["id"])
                
                if config['dataset_name'] == "cpm17":
                    train_data = glob.glob(input_dir_train+'/*.png')
                    test_data = glob.glob(input_dir_test+'/*.png')
                    self.assertEqual(len(train_data),32,"Incorrect number of .png files found! (input_dir_train): "+id)
                    self.assertEqual(len(test_data),32,"Incorrect number of .png files found! (input_dir_test): "+id)
                elif config['dataset_name'] == "kumar":
                    train_data = glob.glob(input_dir_train+'/*.tif')
                    test_data = glob.glob(input_dir_test+'/*.tif')
                    self.assertEqual(len(train_data),16,"Incorrect number of .tif files found! (input_dir_train): "+id)
                    self.assertEqual(len(test_data),14,"Incorrect number of .tif files found! (input_dir_test): "+id)
                else:
                    self.fail("Incorrect dataset name!")
        
        
    def testStatsJSON(self):
        with open('config_stats.json', 'r') as json_file:
            config_list = json.load(json_file)
        
        for config in config_list:
            with self.subTest(i=config["id"]): 
                
                pred_dir_test = config["pred_dir_test"]
                pred_dir_train = config["pred_dir_train"]
                
                true_dir_test = config["true_dir_test"]
                true_dir_train = config["true_dir_train"]
                
                pred_data_test = glob.glob(pred_dir_test+'/*.mat')
                pred_data_train = glob.glob(pred_dir_train+'/*.mat')
                
                true_data_test = glob.glob(true_dir_test+'/*.mat')
                true_data_train = glob.glob(true_dir_train+'/*.mat')
                
                id = str(config["id"])
                
                if config['dataset_name'] == "cpm17":
                    self.assertEqual(len(pred_data_test),32,"Incorrect number of .mat files found! (pred_data_test): "+id)
                    self.assertEqual(len(pred_data_train),32,"Incorrect number of .mat files found! (pred_data_train): "+id)
                    self.assertEqual(len(true_data_test),32,"Incorrect number of .mat files found! (true_data_test): "+id)
                    self.assertEqual(len(true_data_train),32,"Incorrect number of .mat files found! (true_data_train): "+id)
                elif config['dataset_name'] == "kumar":
                    self.assertEqual(len(pred_data_test),14,"Incorrect number of .mat files found! (pred_data_test): "+id)
                    self.assertEqual(len(pred_data_train),16,"Incorrect number of .mat files found! (pred_data_train): "+id)
                    self.assertEqual(len(true_data_test),14,"Incorrect number of .mat files found! (true_data_test): "+id)
                    self.assertEqual(len(true_data_train),16,"Incorrect number of .mat files found! (true_data_train): "+id)
                else:
                    self.fail("Incorrect dataset name!")
            
    
if __name__ == "__main__":
    unittest.main(verbosity=3)
    