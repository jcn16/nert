import os
import cv2

def save_to_video(output_path, output_video_file, frame_rate):
    list_files = os.listdir(output_path)
    all=[]
    for file in list_files:
        all.append(int(file.split('.')[0]))
    all.sort()
    # 拿一张图片确认宽高
    img0 = cv2.imread(os.path.join(output_path, str(all[0])+'.png'))
    # print(img0)
    height, width, layers = img0.shape
    # 视频保存初始化 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videowriter = cv2.VideoWriter(output_video_file, fourcc, frame_rate, (width, height))
    # 核心，保存的东西
    for f in all:
        # print("saving..." + f)
        img = cv2.imread(os.path.join(output_path, str(f)+'.png'))
        videowriter.write(img)
    videowriter.release()
    cv2.destroyAllWindows()
    print('Success save %s!' % output_video_file)

root='/home/jcn/桌面/Nerf_TEST/checkpoints_nert_3/Multi_view'
for i in range(1):
    present_path=os.path.join(root,str(i))
    video_path=os.path.join(root,f'{i}.mp4')

    save_to_video(present_path,video_path,frame_rate=10)