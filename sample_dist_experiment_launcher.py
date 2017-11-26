
#!/usr/bin/python
# Example PBS cluster job submission in Python

import csv
from popen2 import popen2
import time
import argparse
import itertools
import subprocess
import tensorflow as tf

# If you want to be emailed by the system, include these in job_string:
#PBS -M your_email@address
#PBS -m abe  # (a = abort, b = begin, e = end)


def main(FLAGS):

    if tf.gfile.Exists(FLAGS.log_dir):

        tf.gfile.DeleteRecursively(FLAGS.log_dir)

    tf.gfile.MakeDirs(FLAGS.log_dir)

    # Declare experimental flags.
    exp_design = [('rep_num', range(1)),
                  ('train_enqueue_threads', [16]),
                  ('train_batch_size', [16, 32])]

    # Translate the design structure into flag strings.
    exp_flag_strings = [['--' + f + '=' + str(v) for v in r]
                        for (f, r) in exp_design]

    # Produce the Cartesian set of configurations.
    experimental_configs = itertools.product(*exp_flag_strings)

    # Make a list to store job id strings.
    job_ids = []

    # Iterate over each experimental configuration, launching a job for each.
    for i, experimental_config in enumerate(experimental_configs):

        print("-----------------")
        print(experimental_config)
        print("-----------------")

        # Use subproces to command qsub to submit a job.
        p = subprocess.Popen('qsub',
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             shell=True)

        # Customize your options here.
        job_name = "dist_ex_%d" % i
        walltime = "1:00:00"
        select = "1:ncpus=20:mpiprocs=20"
        command = "python ~/tensorflow-zoo/sample_dist_experiment.py"

        # Iterate over flag strings, building the command.
        for flag in experimental_config:

            command += ' ' + flag

        # Add a final flag modifying the log filename to be unique.
        command += ' --log_filename=templog' + str(i)

        # Build the job sting.
        job_string = """#!/bin/bash
        #PBS -N %s
        #PBS -l walltime=%s
        #PBS -l select=%s
        #PBS -o ~/log/output/%s.out
        #PBS -e ~/log/error/%s.err
        #PBS -A MHPCC96670DA1
        #PBS -q standard
        module load anaconda2
        module load tensorflow
        cd $PBS_O_WORKDIR
        %s""" % (job_name, walltime, select, job_name, job_name, command)

        # Print your job string.
        print(job_string)

        # Send job_string to qsub.
        job_ids.append(p.communicate(job_string)[0])

        print("-----------------")

    jobs_complete = False
    timeout = False

    # Loop until timeout or all jobs complete.
    while not(jobs_complete) and not(timeout):

        print("-----------------")

        print('Time elapsed: ' + str(i) + ' seconds.')

        time.sleep(1)

        # Create a list to hold the Bool job complete flags
        job_complete_flags = []

        # Iterate over each job id string.
        for job_id in job_ids:

            # Issue qstat command to get job status.
            p = subprocess.Popen('qstat -r ' + job_id,
                                 stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE,
                                 shell=True)

            # Read the qstat stdout, parse the state, and conv to Boolean.
            job_complete = p.communicate()[0].split()[-2] == 'E'

            # Print a diagnostic.
            print('Job ' + job_id[:-1] + ' complete? ' +
                  str(job_complete) + '.')

            job_complete_flags.append(job_complete)

        timeout = (i > FLAGS.max_runtime)

        jobs_complete = (all(job_complete_flags))

        print("-----------------")

    print("All jobs complete. Merging results")
    # Once all jobs are complete, merge thier outputs.
    # for i, experimental_config in enumerate(experimental_configs):


    # parameter_labels = ['thread_count',
    #                     'batch_size',
    #                     'batch_interval',
    #                     'rep_num',
    #                     'step_num',
    #                     'train_loss',
    #                     'val_loss',
    #                     'mean_running_time']

    # Accomodate Python 3+
    # with open(FLAGS.log_dir '/' + FLAGS.log_filename, 'w') as csvfile:

    # # Accomodate Python 2.7 on Hokulea.
    # with open(FLAGS.log_dir + '/' + FLAGS.log_filename, 'wb') as csvfile:

    #     # Open a writer and write the header.
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerow(parameter_labels)

    #     # Iterate over each output.
    #     for (experimental_configuration, results) in experimental_outputs:

    #         # TODO: Generalize this pattern to not rely on var names.

    #         # Unpack the experimental configuration.
    #         (thread_count,
    #          batch_size,
    #          batch_interval,
    #          rep) = experimental_configuration

    #         # Unpack the cooresponding results.
    #         (steps, train_losses, val_losses, mean_running_times) = results

    #         # Iterate over the results vectors for each config.
    #         for (step, tl, vl, mrt) in zip(steps,
    #                                        train_losses,
    #                                        val_losses,
    #                                        mean_running_times):

    #             # Write the data to a csv.
    #             csvwriter.writerow([thread_count,
    #                                 batch_size,
    #                                 batch_interval,
    #                                 rep,
    #                                 step,
    #                                 tl,
    #                                 vl,
    #                                 mrt])


if __name__ == '__main__':

    # Instantiate an arg parser.
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str,
                        default='../log/sample_dist_experiment/',
                        help='Summaries log directory.')

    parser.add_argument('--log_filename', type=str,
                        default='sample_dist_experiment.csv',
                        help='Merged output filename.')

    parser.add_argument('--max_runtime', type=int,
                        default=360000,
                        help='Number of seconds to run before giving up.')

    # Parse known arguements.
    FLAGS, unparsed = parser.parse_known_args()

    main(FLAGS)
