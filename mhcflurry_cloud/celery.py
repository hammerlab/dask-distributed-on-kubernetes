from __future__ import absolute_import

from celery import Celery

app = Celery('mhcflurry_cloud',
             broker='amqp://',
             backend='amqp://',
             include=['mhcflurry_cloud.model_selection'])

if __name__ == '__main__':
    app.start()