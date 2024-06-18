#!/usr/bin/python
# -*- coding: UTF-8 -*-


import xml.sax
import sys
import os
import pandas as pd
import csv
from os.path import isfile
from Process.utils.utils import get_all_data_files


# 规则函数 1位black 0为white
def Rules(apiname, value):
    # special目录下的exe文件为敏感文件
    special = ['C:', 'C', 'C:/DOCUMENTS AND SETTINGS/ADMIN/LOCAL SETTINGS/APPLICATION DATA/WINDOWS',
               'C:/DOCUMENTS AND SETTINGS/ADMIN/LOCAL SETTINGS/APPLICATION DATA', 'C:/WINDOWS/SYSTEM32',
               'C:/WINDOWS', 'C:/WINDOWS/TEMP', 'C:/DOCUME~1/ADMINI~1/LOCALS~1/TEMP',
               'C:/DOCUMENTS AND SETTINGS/ADMINISTRATOR/LOCAL SETTINGS/TEMP',
               'C:/DOCUMENTS AND SETTINGS/ADMIN/APPLICATION DATA/TEMP',
               'C://PROGRAM', 'C:/DOCUMENTS AND SETTINGS/DEFAULT USER']

    # filepath目录 tempfilename文件名加后缀 filename文件名 extension后缀
    path = value.replace('\\', '/')
    (filepath, tempfilename) = os.path.split(path)
    (filename, extension) = os.path.splitext(tempfilename)

    # 文件路径以及后缀变大写
    filepath = filepath.upper()
    extension = extension.upper()

    # 文件有多个后缀 文件名出现 UuYUIQAM 和 eqgkYooU 这两个字符串
    if filename.find('.') != -1 or filename == 'UuYUIQAM' or filename == 'eqgkYooU':
        return 1

    # C盘、D盘根目录下的单一文件
    if filepath == 'C:' or filepath == 'D:':
        return 1

    # 文件在敏感路径且为exe文件或bat文件等
    if (
            extension == '.EXE' or extension == '.BAT' or extension == '.DLL' or extension == '.SYS' or extension == '.VBS' or extension == '.TMP' or extension == '.DB' or extension == '.LOG' or extension == '.LNK' or extension == '.ICO' or extension == '.MDB') and filepath in special:
        return 1
    # 将scr文件拷贝到system32路径里  将exe文件拷贝到system32的目录下的文件夹或者目录里
    if filepath.find('C:/WINDOWS/SYSTEM32') != -1 and (
            extension == '.SCR' or extension == 'EXE' or extension == 'SYS' or extension == 'DLL' or extension == 'RAR' or extension == 'INI' or extension == 'PDB' or extension == 'RSU'):
        return 1

    if filepath.startswith('C:/CONFIG/DEFAULTSKIN') or filepath.startswith('C:/MSOCACHE') or filepath.startswith(
            'D:/MSOCACHE'):
        return 1

    if (filepath.startswith('C:/DOCUMENTS AND SETTINGS/ADMIN/LOCAL SETTINGS') or filepath.startswith(
            'C:/PROGRAM FILES/COMMON FILES')) and (extension == '.HTM' or extension == '.HTML'):
        return 1

    if filepath.find('C:/PROGRAM FILES') != -1 and (extension == 'HTML' or extension == 'HTM'):
        return 1

    # 删除文件里面中的异常（是否通用？？）
    if filepath.find('C:/DOCUMENTS AND SETTINGS/ALL USERS/DOCUMENTS/MY MUSIC') != -1 and extension == '.WMA':
        return 1
    if (filepath.find('C:/DOCUMENTS AND SETTINGS/ALL USERS/DOCUMENTS/MY PICTURES') != -1 or filepath.find(
            'C:/DOCUMENTS AND SETTINGS/ADMIN/MY DOCUMENTS/MY PICTURES') != -1) and (
            extension == '.JPG' or extension == '.BMP' or extension == '.PNG' or extension == '.DOC' or extension == '.DOCX' or extension == '.XLS' or extension == '.XLSX' or extension == '.TXT'):
        return 1
    if filepath.find('C:/DOCUMENTS AND SETTINGS/ADMIN/MY DOCUMENTS/MY PICTURES') != -1 and (
            extension == '.DOCX' or extension == '.DOC' or extension == '.XLSX' or extension == '.TXT'):
        return 1
    if filepath.find('C:/DOCUMENTS AND SETTINGS/ADMIN/MY DOCUMENTS') != -1 and (
            extension == '.TXT' or extension == '.BMP' or extension == '.JPG' or extension == '.PNG' or extension == '.DOC' or extension == '.XLS'):
        return 1
    if filepath.find('C:/DOCUMENTS AND SETTINGS/ALL USERS/APPLICATION DATA') != -1 and extension == '.BMP':
        return 1
    if filepath.find('C:/DOCUMENTS AND SETTINGS/ADMIN/APPLICATION DATA') != -1 and apiname == 'DeleteFileW':
        return 1
    if filepath.find('C:/PROGRAM FILES') != -1 and (
            extension == '.EXE' or extension == '.DAT') and apiname == 'DeleteFileW':
        return 1

    # 较正常的应用程序安装路径
    if filepath.find('C:/DOCUMENTS AND SETTINGS/ADMIN/APPLICATION DATA') != -1 or filepath.find(
            'C:/PROGRAM FILES') != -1 or filepath.find('C:/PROGRAM') != -1 or filepath.find('C:/PROGRA~1'):
        return 0

    if filepath.startswith('C:/DOCUME~1/ADMINI/LOCALS~1/TEMP') and (
            extension == '.PNG' or extension == '.BMP' or extension == '.JPG' or extension == '.DLL' or extension == '.INI' or extension == '.EXE' or extension == '.LUA' or extension == '.XML' or extension == '.SYS' or extension == '.TMP' or extension == '.BAT' or extension == '.RTF' or extension == '.WXL'):
        if apiname == 'DeleteFileW':
            return 1
        else:
            return 0

    return 2


def OtherRules(value):
    if value.startswith('\\??\\') or value.startswith('\\WINDOWS\\') or value.find('UuYUIQAM') != -1 or value.find(
            'eqgkYooU') != -1 or value.find('SwckEEwQ') != -1 or value.find('aiMoooYw') != -1 or value.find(
        'gegwgsQE') != -1 or value.find('Override') != -1 or value.find('Disable') != -1 or value.find(
        'Enable') != -1 or value.find('Hidden') != -1:
        return 1
    if value.startswith('HKEY_LOCAL_MACHINE\\SOFTWARE\\MICROSOFT') or (
            value.startswith('HKEY_CURRENT_USER') and value.find('SOFTWARE')) or value.startswith(
        'HKEY_LOCAL_MACHINE\\SOFTWARE\\CLASSES\\CLSID'):
        return 1
    if value.startswith('x') and value.find('_'):
        return 1

    if value.startswith('Font #') or value.startswith('Size #') or value.startswith('Color #') or value.startswith(
            '@themeui.dll'):
        return 0
    if value.startswith('HKEY_LOCAL_MACHINE\\SOFTWARE\\MOZILLA') or value.startswith(
            'HKEY_LOCAL_MACHINE\\SOFTWARE\\ODBC') or value.startswith(
        'HKEY_LOCAL_MACHINE\\SOFTWARE\\RISING') or value.startswith(
        'HKEY_LOCAL_MACHINE\\SOFTWARE\\SAMSUNG') or value.startswith(
        'HKEY_LOCAL_MACHINE\\SOFTWARE\\TENCENT') or value.startswith(
        'HKEY_LOCAL_MACHINE\\SOFTWARE\\WISE SOLUTIONS') or value.startswith('HKEY_LOCAL_MACHINE\\SYSTEM'):
        return 0

    ##新增16个API规则
    if value.startswith('/__dmp__/') or value.startswith('/boxstarter/') or value.startswith(
            '/ping.php?') or value.startswith('?opt=put') or value.startswith('/sobaka1.gif?') or value.startswith(
        '/logo.gif?'):
        return 1
    if value.startswith('/') and value.count('/') == 1 and (
            value.endswith('.dat') or value.endswith('.txt') or value.endswith('.doc') or value.endswith(
        '.exe') or value.endswith('.php') or value.endswith('.gif')):
        return 1
    if (value.startswith('/cloud/') and value.endswith('.php')) or (
            value.startswith('/welcom/') and value.endswith('.html')) or (
            value.startswith('/english/') and value.endswith('.php')) or (
            value.startswith('/se/') and value.endswith('.cab')):
        return 1
    if (value.startswith('/install/') or value.startswith('/server/') or value.startswith(
            '/static/')) and value.endswith('.txt'):
        return 1
    if (value.startswith('/images/') or value.startswith('/img/') and (
            value.endswith('.exe') or value.endswith('.jpg') or value.endswith('.gif') or value.endswith(
        '.png'))) or (value.startswith('/upload/') and value.endswith('.jpg')) or (
            value.startswith('/downloads/') or value.startswith('/downloadsoft/') and value.endswith('.exe')):
        return 1

    if (value.startswith('/Download/') or value.startswith('/url/') and value.endswith('.xml')) or (
            value.startswith('/login') or value.startswith('/shendeng') or value.startswith(
        '/dw/') or value.startswith('/tg/') or value.startswith('/manage/') or value.startswith(
        '/projects/bbwin/') or value.startswith('/api/openapi/')):
        return 0
    if value.startswith('/') and value.count('/') == 1 and value.endswith('.swf'):
        return 0
    if value.endswith('/index.htm') or (
            value.startswith('/update/') and (value.endswith('.htm') or value.endswith('.ini'))) or (
            value.startswith('/ad/') and value.endswith('.html')) or (
            value.startswith('/apk/') and value.endswith('.ico')) or (
            value.startswith('/download') and value.endswith('.inf')) or (
            value.startswith('/fieldsync/') and value.endswith('.gz')) or (
            value.startswith('/down/') and value.endswith('.zip')):
        return 0
    if (value.startswith('/down/') or value.startswith('/va/update/') or value.startswith(
            '/uninstall/') or value.startswith('/banner/')) and value.endswith('.htm'):
        return 0

    return 2


def _load_xlsx_data(file_path):
    data = pd.read_excel(file_path)
    # data = data.sort_values('timestamp')
    apis = data['apicall'].tolist()
    pids = data['pid'].tolist()
    _apis2type = data['category'].tolist()
    apis2type = {}
    for api, api2type in zip(apis, _apis2type):
        apis2type[api] = api2type

    # 参数信息
    args = data['args'].tolist()
    # print(args)

    # temp = eval(args[0])
    # print(type(temp), temp)
    # print(type(temp[0]), temp[0])

    return apis, pids, apis2type, args


param_api = ['AutoLogin', 'CopyFileExW', 'CreateProcessInternalW', 'DeleteFileW', 'DeleteService', 'Fake_BeCreatedEx',
             'Fake_BeInjected', 'Fake_SetFileHiddenOrReadOnly', 'Fake_TerminateRemoteProcess', 'gethostbyname',
             'NtAdjustPrivilegesToken', 'NtCreateFile', 'NtCreateThread', 'NtDeleteValueKey', 'NtTerminateProcess',
             'NtTerminateThread', 'NtWriteVirtualMemory', 'SendARP', 'TooManyBehavior', 'TryToAnalyze',
             'WMIConnectServer', 'WMIExecQuery', 'HttpOpenRequestA', 'InternetCrackUrlA', 'InternetCrackUrlW',
             'InternetReadFile', 'URLDownloadToFileW', 'WSAConnect', 'WSASocketW', 'Fake_DetectDebugger',
             'Fake_SetFileCreationTime', 'GetComputerNameExW', 'GetComputerNameW', 'SetWindowsHookExW',
             'StrangeBehavior', 'KiTrap0D', 'LoadLibraryExW', 'MoveFileWithProgressW', 'AnalyzeStart', 'connect',
             'CreateServiceW', 'CryptExportKey', 'CryptGenKey', 'EnumServicesStatusA', 'EnumServicesStatusExW',
             'EnumServicesStatusW', 'Fake_WritePEFile', 'FindFirstFileExW', 'GetAdaptersAddresses',
             'GetForegroundWindow', 'GetProcessAffinityMask', 'KeBugCheck2', 'KiGeneralProtectionFault', 'listen',
             'Login', 'Module32FirstW', 'NtCreateMutant', 'NtCreateThread', 'NtCreateThreadEx', 'NtDeleteFile',
             'NtDeleteKey', 'NtLoadDriver', 'NtQueryAttributesFile', 'NtQueryDirectoryFile',
             'NtQueryFullAttributesFile', 'NtReadFile', 'NtReadVirtualMemory', 'NtSetContextThread',
             'NtSetSystemInformation', 'NtSetSystemTime', 'NtSetValueKey', 'NtShutdownSystem', 'NtSuspendThread',
             'NtUnloadDriver', 'NtWriteFile', 'OpenSCManagerW', 'OpenServiceW', 'PowerOn', 'Process32FirstW', 'recv',
             'recvfrom', 'send', 'sendto', 'StartServiceW', 'Thread32First', 'UnpackSelf', 'VMDetect',
             'WMICreateInstanceEum', 'WSARecv', 'WSARecvFrom', 'WSASend', 'WSASendTo']

# our data
apis = ['RegQueryValueExW', 'WriteFile', 'ExitProcess', 'LoadLibraryA', 'RegOpenKeyExW', 'ReadFile',
        'CreateProcessInternalW', 'CreateFileW', 'VirtualAllocEx', 'OpenMutexW', 'LoadLibraryW', 'RegSetValueExA',
        'RegCreateKeyExW', 'RegOpenKeyExA', 'FindWindowW', 'DeviceIoControl', 'CreateRemoteThread',
        'WriteProcessMemory', 'SetWindowsHookExA', 'ReadProcessMemory', 'RegEnumValueW', 'DeleteFileW',
        'ShellExecuteExW', 'RegSetValueExW', 'CreateMutexW', 'RegEnumKeyExW', 'OpenSCManagerA', 'WinExec',
        'OpenServiceW', 'OpenSCManagerW', 'CopyFileExW', 'RegDeleteKeyA', 'IsDebuggerPresent', 'RegDeleteKeyW',
        'MoveFileWithProgressW', 'CreateServiceA', 'StartServiceW', 'TerminateProcess', 'SetWindowsHookExW',
        'ControlService', 'CreateServiceW', 'DeleteService', 'InternetOpenUrlW', 'URLDownloadToFileW', 'ExitWindowsEx']

api_intersection = ['CopyFileExW', 'MoveFileWithProgressW', 'CreateServiceW', 'CreateProcessInternalW',
                    'OpenServiceW', 'StartServiceW', 'OpenSCManagerW', 'URLDownloadToFileW', 'DeleteService',
                    'SetWindowsHookExW', 'DeleteFileW']


# 统计API的覆盖情况

def api_cover(apis1, apis2=param_api):
    # api覆盖情况
    print("apis len:", len(apis1))
    print("param_api len:", len(apis2))
    print("set(apis1) len:", len(set(apis1)))
    print("set(apis2) len:", len(set(apis2)))
    api_intersection = set(apis1) & set(apis2)
    print("api_intersection len:", len(api_intersection))
    print("api_intersection:", api_intersection)

    return api_intersection


class API_COUNT:
    def __init__(self, api_name):
        self.name = api_name
        self.malious_count = 0
        self.begin_count = 0
        self.unkonw = 0

    def printCount(self):
        print("API:", self.name)
        print("malious_count:", self.malious_count)
        print("begin_count:", self.begin_count)
        print("unkonw:", self.unkonw)


if __name__ == "__main__":
    # ret
    # 0 Malious
    # 1 begin
    # 2 unknow

    # # file_name = data_dir + '4f1fd95a1212f26706c1df6eb3f61d4890f5c37c5de51eb5ee5dfc44cbc44de9.xlsx'

    data_dir = '/root/workspace/yjn/API2Vec/inputs/data/'
    file_paths = get_all_data_files(data_dir)
    print("file paths:", len(file_paths))

    # # 计算API的一个覆盖情况
    # # 读取所有API
    # apis = []
    # for file_name in file_paths:
    #     api_list = _load_xlsx_data(file_name)[0]
    #     api_list_set = set(api_list)
    #     for api_name in api_list_set:
    #         if api_name not in apis:
    #             apis.append(api_name)
    #
    # print("apis:", apis)
    #
    # api_cover(apis)
    #
    # # file_paths_black
    # # len: 14657
    # # file_paths_white
    # # len: 14113
    # # file
    # # paths: 28770
    # # apis: ['RegQueryValueExW', 'CreateProcessInternalW', 'ReadFile', 'VirtualAllocEx', 'RegOpenKeyExW', 'OpenMutexW',
    # #        'LoadLibraryA', 'CreateFileW', 'WriteFile', 'ExitProcess', 'WriteProcessMemory', 'LoadLibraryW',
    # #        'FindWindowW', 'DeviceIoControl', 'RegCreateKeyExW', 'RegOpenKeyExA', 'RegSetValueExA', 'CreateRemoteThread',
    # #        'SetWindowsHookExA', 'DeleteFileW', 'RegSetValueExW', 'ShellExecuteExW', 'CreateMutexW', 'RegEnumValueW',
    # #        'ReadProcessMemory', 'RegEnumKeyExW', 'WinExec', 'OpenSCManagerW', 'OpenSCManagerA', 'OpenServiceW',
    # #        'CopyFileExW', 'RegDeleteKeyA', 'IsDebuggerPresent', 'RegDeleteKeyW', 'MoveFileWithProgressW',
    # #        'StartServiceW', 'CreateServiceA', 'TerminateProcess', 'SetWindowsHookExW', 'ControlService',
    # #        'CreateServiceW', 'DeleteService', 'InternetOpenUrlW', 'URLDownloadToFileW', 'ExitWindowsEx']
    # # apis
    # # len: 45
    # # param_api
    # # len: 92
    # # api_intersection
    # # len: 11
    # # api_intersection: {'CopyFileExW', 'MoveFileWithProgressW', 'CreateServiceW', 'CreateProcessInternalW',
    # #                    'OpenServiceW', 'StartServiceW', 'OpenSCManagerW', 'URLDownloadToFileW', 'DeleteService',
    # #                    'SetWindowsHookExW', 'DeleteFileW'}
    #
    #

    # 参数覆盖情况

    # 统计参数铺盖情况
    m_count, b_count, u_count = 0, 0, 0
    api_count = {}
    i = 0
    for file_name in file_paths[-8:]:
        i += 1
        print(file_name)
        apis, pids, apis2type, args = _load_xlsx_data(file_name)

        # print("apis len:", len(apis))

        apis_counts = {}
        for api_name in apis:
            if api_name in apis_counts:
                continue
            else:
                apis_counts[api_name] = API_COUNT(api_name)

        for api_name, api_args in zip(apis, args):
            api_args_dict_list = eval(api_args)
            # print(api_args_dict)
            retnum = []
            for arg_name_value in api_args_dict_list:
                arg_name = arg_name_value['name']
                value = arg_name_value['value']
                # print("name:", api_name)
                # print("value:", value)
                if value.find('C:\\') != -1 or value.find('c:\\') != -1 or value.find('D:\\') != -1 or value.find(
                        'd:\\') != -1 or value.find('E:\\') != -1 or value.find('e:\\') != -1:
                    # print("Rules")
                    retnum.append(Rules(api_name, value))
                else:
                    # print("OtherRules")
                    retnum.append(OtherRules(value))

            # 对于每个api来说 更新
            if 0 in retnum:
                apis_counts[api_name].malious_count += 1
            elif 1 in retnum:
                apis_counts[api_name].begin_count += 1
            else:
                apis_counts[api_name].unkonw += 1

        api_names = []
        malious_counts = []
        begin_counts = []
        unkonws = []
        files = []
        for api_name, api_count in apis_counts.items():
            api_names.append(api_count.name)
            malious_counts.append(api_count.malious_count)
            begin_counts.append(api_count.begin_count)
            unkonws.append(api_count.unkonw)
            files.append(file_name)

        write_content = pd.DataFrame(
            {'file_name': files, 'api_name': api_names, 'malious_count': malious_counts,
             'begin_count': begin_counts, 'unkonw_count': unkonws
             })
        write_content.to_csv('./white_malious_begin_unkonw_count.csv', mode='a', encoding='gbk')
