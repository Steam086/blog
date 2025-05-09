
### 关于Fedora上的samba服务器无法被访问，显示access denied

Fedora的默认设置是
```bash
sudo getenforce
> Enforcing
```
输入
```bash
sudo setenforce 0
```
可以将模式改为Permissive，此时samba服务器可以正常被访问
```bash
sudo getenforce
Permissive
```

### 还是不行怎么办

输入上述指令之后，在本机的网络中可以成功访问，但是在局域网中的设备，既无法ping通也没法连接到smb，猜测是防火墙相关设置拦住了icmp，询问GPT解决方案，使用如下命令
```bash
sudo firewall-cmd --zone=trusted --permanent --add-service=samba  
sudo firewall-cmd --reload
```
我们连接的网络里可以设置firewall zone，将firewall zone设置为trusted可以生效

------
解决方式：
询问kimi，kimi有如下解释
### 5. **SELinux限制**

- 如果SELinux处于强制模式，可能会阻止Samba访问某些目录。可以尝试临时将SELinux设置为宽容模式来测试：

    ```bash
    sudo setenforce 0
    ```
- 如果问题解决，说明SELinux是导致问题的原因。可以为共享目录设置正确的上下文：
    ```bash
    sudo chcon -t samba_share_t /path/to/your/folder
    ```

