import os
import logging

def createFile(filePath):
    if os.path.exists(filePath):
        print('logger path:{}'.format(filePath))
    else:
        os.makedirs(filePath)
        print('create logger path:{}'.format(filePath))

# logging.StreamHandler	将日志消息发送到输出到Stream，如std.out, std.err或任何file-like对象。
# logging.FileHandler	将日志消息发送到磁盘文件，默认情况下文件大小会无限增长
# logging.handlers.RotatingFileHandler	将日志消息发送到磁盘文件，并支持日志文件按大小切割
# logging.hanlders.TimedRotatingFileHandler	将日志消息发送到磁盘文件，并支持日志文件按时间切割
# logging.handlers.HTTPHandler	将日志消息以GET或POST的方式发送给一个HTTP服务器
# logging.handlers.SMTPHandler	将日志消息发送给一个指定的email地址
# logging.NullHandler	该Handler实例会忽略error messages，通常被想使用logging的library开发者使用来避免'No handlers could be found for logger XXX'信息的出现。

root_path = './data/logs/'
createFile(root_path)

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'incremental': False,
    'formatters': {
        'standard': {
            'class': 'logging.Formatter',
            'format': '%(asctime)s [%(threadName)s] [%(levelname)s] %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
    },
    'filters': {},
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'standard',
        },
        # 'error': {
        #     'level': 'DEBUG',
        #     'class': 'logging.handlers.TimedRotatingFileHandler',
        #     'formatter': 'standard',
        #     'filename':os.path.join(root_path, '/error.log'),
        #     'interval': 1,
        #     'when': 'D',
        #     'backupCount': 999,
        #     'encoding': 'utf-8',
        # },
        'info': {
            'level': 'DEBUG',
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'formatter': 'standard',
            'filename': os.path.join(root_path, 'info.log'),
            'interval': 1,
            'when': 'D',
            'backupCount': 999,
            'encoding': 'utf-8',
        },
    },
    'loggers': {
        'console': {
            'handlers': ['console','info'],
            'level': 'DEBUG',
            'propagate': True
        },
        # 'error': {
        #     'handlers': ['error'],
        #     'level': 'DEBUG',
        #     'propagate': True
        # },
        'info': {
            'handlers': ['info'],
            'level': 'INFO',
            'propagate': True
        }
    }
}



import logging.config as config
config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('console')


if __name__=='__main__':
  import logging.config as config
  config.dictConfig(LOGGING_CONFIG)
  logger.info('kkkk')
  logger.error('kkkk')
