import torch


class nan_prosess_Layer(torch.autograd.Function):
    #
    # def __init__():
    #     pass

    @staticmethod
    def forward(ctx, input, logger):
        with torch.no_grad():
            if True in torch.isinf(input):
                logger.info('forward inf_for_svd')
                input = torch.nan_to_num(input)
                if True in torch.isinf(input):
                    logger.info('forward inf_to_num fails')
                else:
                    logger.info('forward inf_to_num success')
            if True in torch.isnan(input):
                logger.info('forward nan_for_svd')
                input = torch.nan_to_num(input)
                if True in torch.isnan(input):
                    logger.info('forward nan_to_num fails')
                else:
                    logger.info('forward nan_to_num success')
        return input

    @staticmethod
    def backward(ctx, gradOutput):
        # from main import log_dir
        # logger = set_logger(log_dir)
        # logger.info('log_dir: {}'.format(log_dir))
        with torch.no_grad():
            if True in torch.isinf(gradOutput):
                from main import logger as logger1
                gradOutput = torch.nan_to_num(gradOutput)
                if True in torch.isinf(gradOutput):
                    logger1.info('backbard inf_to_num fails')
                else:
                    logger1.info('backbard inf_to_num success')
            if True in torch.isnan(gradOutput):
                from main import logger as logger1
                gradOutput = torch.nan_to_num(gradOutput)
                if True in torch.isnan(gradOutput):
                    logger1.info('backbard nan_to_num fails')
                else:
                    logger1.info('backbard nan_to_num success')
        return gradOutput, None
