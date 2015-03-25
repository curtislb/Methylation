project 2, methylation data imputation
EDIT: March 13/2015: please download new data which corresponds to less NaN
entries. The names of the form intersected_final_chr1_cutoff_20_train.bed
are NOW: intersected_final_chr1_cutoff_20_train_revisted.bed

bed file format for k processed, aggregated samples
- tab separated

column          short name                     description      
1                    chromosome                  chromosome of methylated site
2                    start position                  position of the methylated C (inclusive)
3                    end position                   position of the G following the methylated C (exclusive)
4                    strand                             + or - depending on whether it is on 5’-3’ or 3’-5'
5 - (5+k-1)     methylation values calls  corrected methylation value for each sample
5 + k               hg                                   0 or 1, 1 if the position defined by columns 1,2,3 is
                                                             present on the illumine 450K chip

 
For 22 chromosomes, you will be given 3 files

train.bed

The training data contains the aggregated information from 32 samples (corresponding to columns 5,6,..31).
These samples correspond to different tissues/ conditions in which the data has been collected.
The format is that of a bed file for k = 32 processed, aggregated samples.

sample.bed

The test data contains partial information from a sample not present in the training data. This data has the following format:

- tab separated

column          short name                     description      
1                    chromosome                  chromosome of methylated site
2                    start position                  position of the methylated C (inclusive)
3                    end position                   position of the G following the methylated C (exclusive)
4                    strand                             + or - depending on whether it is on 5’-3’ or 3’-5'
5             methylation values calls        nan - if the value on column 6 is 1, the corrected methylation value otherwise                                                                
6                    450K_chip                      0 or 1; 1 if the position defined by columns 1,2,3 is
                                                             present on the illumina 450K chip

Note 0: Recall that the goal of this project is to use the training data (train.bed) and the new sample information (sample.bed) to “impute” or fill in the missing values in column 5 of sample.bed
To evaluate your prediction you will make use of the test.bed file which contains information about the “true” methylation value for all positions.

test.bed

- tab separated

column          short name                     description      
1                    chromosome                  chromosome of methylated site
2                    start position                  position of the methylated C (inclusive)
3                    end position                   position of the G following the methylated C (exclusive)
4                    strand                             + or - depending on whether it is on 5’-3’ or 3’-5'
5             methylation values calls        corrected methylation value call                                                               
6                    450K_chip                          0 or 1; 1 if the position defined by columns 1,2,3 is
                                                             present on the illumina 450K chip

Note 00: Even though you can run your algorithms on all the chromosomes. We recommend  you use data from chromosomes 1, 2, 6, 7, 11with a cutoff of 20. Namely the files:
intersected_final_chr1_cutoff_20_sample.bed; intersected_final_chr1_cutoff_20_train.bed; intersected_final_chr1_cutoff_20_test.bed;
 
intersected_final_chr2_cutoff_20_sample.bed; intersected_final_chr2_cutoff_20_train.bed; intersected_final_chr2_cutoff_20_test.bed; 

intersected_final_chr6_cutoff_20_sample.bed; intersected_final_chr6_cutoff_20_train.bed; intersected_final_chr6_cutoff_20_test.bed; 

intersected_final_chr7_cutoff_20_sample.bed; intersected_final_chr7_cutoff_20_train.bed; intersected_final_chr7_cutoff_20_test.bed; 

intersected_final_chr11_cutoff_20_sample.bed; intersected_final_chr11_cutoff_20_train.bed; intersected_final_chr11_cutoff_20_test.bed; 

For more explanation see below:

 
Note 1: the names of the files you will see in this folder are of the form:
intersected_final_chr2_cutoff_10_sample.bed; intersected_final_chr2_cutoff_10_train.bed; intersected_final_chr2_cutoff_10_test.bed; 

The intersected_final_chr2_cutoff_10_ part comes from the processing pipeline. “intersected_final” means that the file is aggregated
“chr2” means that the data comes from chromosome 2
“cutoff_10_” means that a quality control has been performed such that across all the positions collected (for every row) at least 90 (= 100 - 10)% of the values for the 32 samples are well defined (not nan).

Note 2: Even given the quality control, there are still entries in the training data that contain ‘nan’ values. Think what would be appropriate to do in those cases and how would that affect your final prediction.



