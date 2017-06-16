import sugartensor as tf
from functools import wraps
from multiprocessing import Process
import time
import zmq
import msgpack
import msgpack_numpy
from sugartensor.sg_queue import _FuncQueueRunner


# enable msgpack to handle numpy
msgpack_numpy.patch()


def sg_ventilator(ip='127.0.0.1', has_worker=False, hwm=8192, epoch=-1):

    def decorator(func):

        @wraps(func)
        def wrapper(**kwargs):

            # worker for process
            def _worker():

                # port #
                port = 5556 if has_worker else 5557

                # url for ventilator queue
                url_ = 'tcp://%s:%d' % (ip, port)

                # queue for ventilator
                ctx = zmq.Context()
                vent = ctx.socket(zmq.PUSH)
                vent.set_hwm(hwm)

                # bind or connect
                if has_worker:
                    vent.bind(url_)
                else:
                    vent.connect(url_)

                # loop for each epoch
                cnt = epoch if epoch > 0 else 1000000
                for _ in range(cnt):
                    # call ventilator function to get data
                    for data_ in func(tf.sg_opt(kwargs)):
                        # sent to worker or sinker
                        vent.send(msgpack.packb(data_), copy=False)
                        time.sleep(1e-3)

            # start worker as new process
            p = Process(target=_worker, args=())
            p.daemon = True
            p.start()

        return wrapper
    return decorator


def sg_worker(vent_ip='127.0.0.1', sink_ip='127.0.0.1', worker_nums=1, sink_hwm=8192, vent_hwm=8192):

    def decorator(func):

        @wraps(func)
        def wrapper(**kwargs):

            # worker for process
            def _worker():

                # url for ventilator queue
                vent_url_ = 'tcp://%s:%d' % (vent_ip, 5556)

                # url for sinker queue
                sink_url_ = 'tcp://%s:%d' % (sink_ip, 5557)

                # queue for ventilator
                ctx = zmq.Context()
                vent = ctx.socket(zmq.PULL)
                vent.set_hwm(vent_hwm)
                vent.connect(vent_url_)

                # queue for sinker
                ctx = zmq.Context()
                sink = ctx.socket(zmq.PUSH)
                sink.set_hwm(sink_hwm)
                sink.connect(sink_url_)

                # endless loop
                while True:
                    # get data from ventilator
                    data_ = msgpack.unpackb(vent.recv(copy=False).bytes)
                    # call function
                    processed = func(tf.sg_opt(kwargs), *data_)
                    # sent to sinker
                    sink.send(msgpack.packb(processed), copy=False)
                    time.sleep(1e-3)

            # start workers as new process
            for _ in range(worker_nums):
                p = Process(target=_worker, args=())
                p.daemon = True
                p.start()

        return wrapper
    return decorator


def sg_sinker(ip='127.0.0.1', hwm=8192):

    def decorator(func):

        @wraps(func)
        def wrapper(**kwargs):

            # url for sinker
            url_ = 'tcp://%s:%d' % (ip, 5557)

            # queue for sink
            ctx = zmq.Context()
            sink = ctx.socket(zmq.PULL)
            sink.set_hwm(hwm)
            sink.bind(url_)

            # get data
            while True:
                # get data from sinker queue
                data_ = msgpack.unpackb(sink.recv(copy=False).bytes)
                # call function
                yield func(tf.sg_opt(kwargs), *data_)
                time.sleep(1e-3)

        return wrapper
    return decorator


def sg_tf_sinker(dtypes, capacity=8192, ip='127.0.0.1', hwm=8192):

    # url for sinker queue
    url_ = 'tcp://%s:%d' % (ip, 5557)

    # queue for sink
    ctx = zmq.Context()
    sink = ctx.socket(zmq.PULL)
    sink.set_hwm(hwm)
    sink.bind(url_)

    # create place holder
    phs_ = []
    for dt in dtypes:
        phs_.append(tf.placeholder(dtype=dt))

    # enqueue function
    def enqueue_func(sess, op):

        # get data from sink
        data_ = msgpack.unpackb(sink.recv(copy=False).bytes)

        # run session
        dic_ = {}
        for p, d in zip(phs_, data_):
            dic_[p] = d
        sess.run(op, feed_dict=dic_)

    # create FIFO queue
    queue = tf.FIFOQueue(capacity, dtypes)

    # enqueue operation
    enqueue_op = queue.enqueue(phs_)

    # create queue runner
    runner = _FuncQueueRunner(enqueue_func, queue, [enqueue_op])

    # register to global collection
    tf.train.add_queue_runner(runner)

    # return de-queue operation
    return queue.dequeue()

