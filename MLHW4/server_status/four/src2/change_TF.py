import csv


if __name__ == "__main__":
    

    input_csv = "./submission/ensemble_b5_b5_pseudo_b6.csv"
    output_csv = "./submission/ensemble_b5_b5_pseudo_b6.csv"

    rows = []

    # 讀取 CSV
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)   # 讀欄位名稱（filename, label）

        for filename, label in reader:
            # 將 True/False → 1/0
            if label == "True":
                new_label = "1"
            else:               # False 或其他
                new_label = "0"
            rows.append([filename, new_label])

    # 寫回新的 CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"✔ 已轉換並輸出：{output_csv}")
