"""
测试所有故障的检测性能
"""
import warnings
warnings.filterwarnings('ignore')
import subprocess
import re
import pandas as pd
import os

def test_fault(fault_num):
    """测试指定故障"""
    # 修改fault_detection1.py中的FAULT_NUMBER
    with open('src/fault_detection1.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = re.sub(r'FAULT_NUMBER = \d+', f'FAULT_NUMBER = {fault_num}', content)
    
    with open('src/fault_detection1.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    # 运行测试
    result = subprocess.run(
        ['python', 'src/fault_detection1.py'],
        capture_output=True,
        text=True
    )
    
    # 解析结果
    output = result.stdout
    
    FAR = None
    FDR = None
    
    for line in output.split('\n'):
        if '正常数据误报率' in line:
            try:
                FAR = float(line.split(':')[1].strip().replace('%', ''))
            except:
                pass
        if '故障数据检出率' in line:
            try:
                FDR = float(line.split(':')[1].strip().replace('%', ''))
            except:
                pass
    
    return FAR, FDR


def main():
    """测试所有故障"""
    
    print("="*80)
    print("测试所有故障的检出率")
    print("="*80)
    
    # 测试所有故障
    results = []
    
    for fault_num in range(1, 22):
        print(f"\n测试故障 {fault_num:02d}...", end=" ")
        
        FAR, FDR = test_fault(fault_num)
        
        if FAR is not None and FDR is not None:
            status = '✓' if (FAR <= 2.0 and FDR >= 90.0) else '✗'
            print(f"FAR={FAR:.2f}%, FDR={FDR:.2f}% {status}")
            
            results.append({
                'Fault': fault_num,
                'FAR': FAR,
                'FDR': FDR,
                'Status': status
            })
        else:
            print(f"解析失败")
            results.append({
                'Fault': fault_num,
                'FAR': None,
                'FDR': None,
                'Status': '?'
            })
    
    # 保存结果
    df = pd.DataFrame(results)
    
    if not os.path.exists('results'):
        os.makedirs('results')
    
    df.to_csv('results/all_faults_test.csv', index=False)
    
    # 显示汇总
    print("\n" + "="*80)
    print("汇总结果")
    print("="*80)
    print(df.to_string(index=False))
    
    # 统计
    达标数 = len(df[df['Status'] == '✓'])
    总数 = len(df)
    
    print(f"\n达标故障数: {达标数}/{总数}")
    print(f"达标率: {达标数/总数*100:.1f}%")
    
    print(f"\n✓ 结果已保存到 results/all_faults_test.csv")


if __name__ == "__main__":
    main()

