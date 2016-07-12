# mhcflurry-cloud
Infrastructure for mhcflurry model selection in the cloud

## Running locally

Install the package:
```
$ pip install -e .
```

We'll run locally using RabbitMQ. See [docs](http://docs.celeryproject.org/en/latest/getting-started/brokers/rabbitmq.html#broker-rabbitmq).

Setup rabbitmq:
```
$ brew install rabbitmq
$ ln -sfv /usr/local/opt/rabbitmq/*.plist ~/Library/LaunchAgents  # launch on startup
$ launchctl load ~/Library/LaunchAgents/homebrew.mxcl.rabbitmq.plist  # start now
```

Start celery worker:
```
$ celery -A mhcflurry_cloud.model_selection.app worker --loglevel=info --concurrency=10
```

Try the example:
```
$ python example.py
```