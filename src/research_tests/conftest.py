import pytest

CV_NUM = 10

# ##########------------  GPU ASSIGNMENT FIXTURE  -------------###########


def pytest_addoption(parser):
    parser.addoption("--gpus", type=str, help="comma seperated list of available gpus", required=True)


def pytest_generate_tests(metafunc):
    if "gpus" in metafunc.fixturenames:
        metafunc.parametrize("gpus", [metafunc.config.getoption("gpus")])


@pytest.fixture()
def gpu(worker_id, gpus):
    gpus_list = gpus.split(',')
    worker_num = int(worker_id[2:])
    return gpus_list[worker_num % len(gpus_list)]
    # We allow for using same gpu by different processes, for cpu intensive tasks
    # try:
    #     return gpus_list[worker_num]
    # except IndexError:
    #     raise Exception('Number of workers needs to be smaller or equal to number of gpus available. No gpu for '
    #                     'worker {}.'.format(worker_id))


# Cross validation fixture
cv_list = list(range(CV_NUM))
@pytest.fixture(params=cv_list)
def cv(request):
    return request.param


# Split name and ds name fixture
split_ds_names_list = [
    ('work1gen1reg', 'pokec_n'),
    ('work1gen1reg', 'pokec_z'),
    ('salary1age4cou', 'nba'),
]
@pytest.fixture(params=split_ds_names_list)
def split_ds_names(request):
    return request.param


# Number of epochs fixture
epochs_list = [500]
@pytest.fixture(params=epochs_list)
def epochs(request):
    return request.param
