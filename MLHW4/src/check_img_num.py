import os
from pathlib import Path

def count_jpg_files(folder_path, recursive=False):
    """
    計算資料夾內的 .jpg 檔案數量
    
    參數:
        folder_path: 資料夾路徑
        recursive: 是否遞迴搜尋子資料夾 (預設 False)
    
    回傳:
        JPG 檔案數量
    """
    if not os.path.exists(folder_path):
        print(f"錯誤: 路徑 '{folder_path}' 不存在")
        return 0
    
    if not os.path.isdir(folder_path):
        print(f"錯誤: '{folder_path}' 不是資料夾")
        return 0
    
    jpg_extensions = {'.jpg', '.jpeg', '.JPG', '.JPEG'}
    count = 0
    
    if recursive:
        # 遞迴搜尋所有子資料夾
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if Path(file).suffix in jpg_extensions:
                    count += 1
    else:
        # 只搜尋當前資料夾
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path) and Path(item).suffix in jpg_extensions:
                count += 1
    
    return count


if __name__ == "__main__":
    # 使用方式 1: 指定資料夾路徑
    folder = input("請輸入資料夾路徑 (直接按 Enter 使用當前目錄): ").strip()
    
    if not folder:
        folder = "."
    
    # 詢問是否要搜尋子資料夾
    recursive_input = input("是否搜尋子資料夾? (y/n, 預設 n): ").strip().lower()
    recursive = recursive_input == 'y'
    
    # 計算數量
    count = count_jpg_files(folder, recursive)
    
    if recursive:
        print(f"\n'{folder}' 及其子資料夾中共有 {count} 個 JPG 檔案")
    else:
        print(f"\n'{folder}' 中共有 {count} 個 JPG 檔案")