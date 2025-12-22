import torch
import sys

def check_flash_attn():
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    
    # 1. 检查 CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA is NOT available. Flash Attention requires GPU.")
        return False
    
    device = torch.device("cuda")
    print(f"CUDA version (torch): {torch.version.cuda}")
    
    # 2. 检查算力 (Compute Capability)
    # Flash Attention 2 需要 Ampere (8.0) 或更高架构
    capability = torch.cuda.get_device_capability(device)
    major, minor = capability
    cc_str = f"{major}.{minor}"
    print(f"GPU: {torch.cuda.get_device_name(device)} (Compute Capability: {cc_str})")
    
    if major < 8:
        print(f"⚠️  Warning: Flash Attention 2 requires Compute Capability >= 8.0 (Ampere).")
        print(f"    Your GPU ({cc_str}) might only support Flash Attention 1.x or standard attention.")
    
    # 3. 尝试导入 Flash Attention
    try:
        import flash_attn
        print(f"✅ Flash Attention package found. Version: {flash_attn.__version__}")
        
        # 尝试导入具体函数以确保编译无误
        from flash_attn import flash_attn_func
        print("✅ flash_attn_func loaded successfully.")
        
        # 检查是否支持 v2
        if int(flash_attn.__version__.split('.')[0]) >= 2:
            print("🚀 Ready for Flash Attention 2!")
            return True
        else:
            print("⚠️  Installed version is < 2.0. Recommended to upgrade.")
            return False
            
    except ImportError:
        print("❌ Flash Attention package NOT found.")
        return False
    except Exception as e:
        print(f"❌ Flash Attention found but failed to load. Error:\n{e}")
        return False

if __name__ == "__main__":
    print("-" * 30)
    success = check_flash_attn()
    print("-" * 30)
    if success:
        print("结论: 你的环境支持并已配置 Flash Attention 2。")
    else:
        print("结论: 需要安装或修复 Flash Attention。")