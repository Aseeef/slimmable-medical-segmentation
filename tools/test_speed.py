import sys, time, torch
from os import path

from configs import USSlimDuckNetConfig

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from models import get_model


def test_model_speed(config, ratio=0.5, imgw=1920, imgh=1080, iterations=None, model_width_mult=None):
    # Codes are based on 
    # https://github.com/ydhongHIT/DDRNet/blob/main/segmentation/DDRNet_23_slim_eval_speed.py

    if ratio != 1.0:
        assert ratio > 0, 'Ratio should be larger than 0.\n'
        imgw = int(imgw * ratio)
        imgh = int(imgh * ratio)

    device = torch.device('cuda')
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True

    model = get_model(config)
    if model_width_mult is not None:
        model.apply(lambda m: setattr(m, 'width_mult', model_width_mult))
    model.eval()
    model.to(device)
    print('\n=========Speed Testing=========')
    print(f'Model: {config.model}\nEncoder: {config.encoder}\nDecoder: {config.decoder}')
    print(f'Size (W, H): {imgw}, {imgh}')

    input = torch.randn(1, 3, imgh, imgw).cuda()
    with torch.no_grad():
        for _ in range(10):
            model(input)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(f'FPS: {FPS}\n')


if __name__ == '__main__':
    config = USSlimDuckNetConfig()
    config.init_dependent_config()

    config.use_aux = False

    for w in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        test_model_speed(config, model_width_mult=w)