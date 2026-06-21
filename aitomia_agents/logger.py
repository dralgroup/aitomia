import logging
import sys
from .settings import settings
# from .settings import settings
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.MAGENTA + Style.BRIGHT,
    }
    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        message = super().format(record)
        return f"{color}{message}{Style.RESET_ALL}"

class Logger:
    
    def __init__(self, name: str = "aitomia"):
        self.logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self):
        log_level = getattr(settings, 'LOG_LEVEL', 'DEBUG').upper()
        self.logger.setLevel(getattr(logging, log_level))
        self.logger.handlers.clear()
        
        if hasattr(settings, 'log_file') and settings.log_file:
            import os 
            # 展开 ~ 为用户主目录
            log_file_path = os.path.expanduser(settings.log_file)
            
            if settings.log_file:
                # 确保日志目录存在
                log_dir = os.path.dirname(log_file_path)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                
                if os.path.exists(log_file_path):
                    os.remove(log_file_path)
                file_handler = logging.FileHandler(log_file_path)
                file_handler.setLevel(getattr(logging, log_level))
                # 文件日志不加颜色
                file_formatter = logging.Formatter(
                    fmt=getattr(settings, 'LOG_FORMAT', "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
                    datefmt="%Y-%m-%d %H:%M:%S"
                )
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
        else:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level))
            formatter = ColorFormatter(
                fmt=getattr(settings, 'LOG_FORMAT', "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def get_logger(self) -> logging.Logger:
        """获取配置好的日志器"""
        return self.logger
    
    def info(self, message: str, *args, **kwargs):
        """记录信息级别日志"""
        self.logger.info(message, *args, **kwargs)
    
    def debug(self, message: str, *args, **kwargs):
        """记录调试级别日志"""
        self.logger.debug(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """记录警告级别日志"""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """记录错误级别日志"""
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """记录严重错误级别日志"""
        self.logger.critical(message, *args, **kwargs)


# 创建全局日志器实例
logger = Logger()

# 导出常用接口
def get_logger() -> logging.Logger:
    """获取日志器实例"""
    return logger.get_logger()

def log_info(message: str, *args, **kwargs):
    """记录信息日志"""
    logger.info(message, *args, **kwargs)

def log_debug(message: str, *args, **kwargs):
    """记录调试日志"""
    logger.debug(message, *args, **kwargs)

def log_warning(message: str, *args, **kwargs):
    """记录警告日志"""
    logger.warning(message, *args, **kwargs)

def log_error(message: str, *args, **kwargs):
    """记录错误日志"""
    logger.error(message, *args, **kwargs)

def log_critical(message: str, *args, **kwargs):
    """记录严重错误日志"""
    logger.critical(message, *args, **kwargs)