from __future__ import absolute_import

from celery import Celery

app = Celery('mhcflurry_cloud',
             broker='amqp://',
             backend='amqp://',

             # This list must include all modules that define celery tasks:
             include=[
                'mhcflurry_cloud.train'])

if __name__ == '__main__':
    app.start()