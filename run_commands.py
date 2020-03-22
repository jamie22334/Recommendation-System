import json
import time
import subprocess, threading
import os, signal, sys


def generate_python_command(input_path, name, data_path, result_path, spark_dir, log_path):

    main_file = os.path.join(input_path, name + '.py')
    test_file = os.path.join(data_path, 'yelp_val.csv')
    output_file = os.path.join(result_path, name + '.csv')
    log_file = os.path.join(log_path, name + '.log')

    return '{}spark-submit --executor-memory 4G --driver-memory 4G {} {} {} {} >{} 2>&1' \
            .format(spark_dir, main_file, data_path, test_file, output_file, log_file)


class Command(object):
    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None

    def run(self, timeout):
        def target():
            self.process = subprocess.Popen(self.cmd, shell=True, preexec_fn=os.setsid)
            self.process.communicate()

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(timeout)
        if thread.is_alive():
            print('Terminating process.', self.cmd.split(' ')[-2].split('/')[-1].split('.')[0])
            os.killpg(self.process.pid, signal.SIGTERM)
            thread.join()


if __name__ == '__main__':

    setting_file = sys.argv[1]
    json_data = open(setting_file).read()

    # json_data = open('data/settings2.json').read()

    settings = json.loads(json_data)

    target_path = settings['target_path']
    spark_dir = settings['spark_dir']
    data_path = settings['data_path']
    result_path = settings['result_path']
    result_log_path = settings['result_log_path']

    student_name = os.path.basename(target_path)

    os.chdir(target_path)
    command = generate_python_command(target_path, student_name, data_path, result_path, spark_dir, result_log_path)
    print(command)

    try:
        start_time = time.time()
        command_thred = Command(command)
        command_thred.run(timeout=60 * 20)
        print('Finished {} in {}'.format(student_name, time.time() - start_time))

    except Exception as e:
        print('Failed {}'.format(student_name))

