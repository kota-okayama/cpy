"""Notification"""

import os

import yaml
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.header import Header

from .utils import path
from .datatype.workflow import WorkflowConfig
from .logger import Logger, LoggerConfig

# 参考
# https://zenn.dev/eito_blog/articles/af53c6b4390ee9
# https://zenn.dev/eito_blog/articles/8c97f0bcbc3260


class GmailNotification:
    """メール通知を行うクラス"""

    @classmethod
    def send(cls, project_dirpath: str, config: WorkflowConfig):
        """[classmethod] 現状のワークフローについてGメールを送信する"""

        subject = ""  # 件名
        body = ""  # 本文
        title = ""
        project_name = (
            path.abspath(project_dirpath).removeprefix(f"{path.abspath('@')}/")
            if path.abspath(project_dirpath).startswith(f"{path.abspath('@')}/project")
            else path.basename(project_dirpath)
        )

        # ワークフローのHTMLを作成
        metrics = config.metrics.format_for_notifications(config.current_workflow)
        workflow = ""
        for i, w in enumerate(config.workflow):
            workflow += (
                f"<li style='margin-left:1em;'><b>{w['name']}</b>{metrics.workflow_time[i]}</li>"
                if i == config.current_workflow
                else f"<li>{w['name']}{metrics.workflow_time[i]}</li>"
            )

        # 件名と本文の生成
        if config.current_workflow == len(config.workflow):
            subject = "[Finished] TriWaveワークフローが完了しました"
            title = "TriWaveにより処理中のワークフローが完了しました"

        else:
            subject = "[Suspend] TriWaveワークフローが中断しました"
            title = "TriWaveにより処理中のワークフローが中断しました"

        body = f"""
<html>
    <head></head>
    <body>
        <div>{title}</div>
        <div>結果を確認してください</div><br>
        <div>プロジェクト名: <b>{project_name}</b></div>
        <div>実行時間: {metrics.execution_time}</div>
        <div>ワークフロー:</div>
        <ol start='0'>
            {workflow}
        </ol>
    </body>
</html>
        """

        charset = "iso-2022-jp"
        msg = MIMEMultipart()
        msg.attach(MIMEText(body, "html", charset))
        msg["Subject"] = Header(subject.encode(charset), charset)

        with open(path.join(project_dirpath, config.log_filepath), "rb") as f:
            mb = MIMEApplication(f.read())

        mb.add_header("Content-Disposition", "attachment", filename="log.txt")
        msg.attach(mb)

        smtp_obj = smtplib.SMTP("smtp.gmail.com", 587)
        smtp_obj.ehlo()
        smtp_obj.starttls()
        smtp_obj.login(config.gmail_sender, config.gmail_app_pw)

        # メール送信
        Logger(
            __name__,
            logger_config=LoggerConfig(level=config.log_level),
            filepath=path.join(project_dirpath, config.log_filepath),
        ).info("Start sending Gmail.")
        print("Start sending Gmail.")

        smtp_obj.sendmail(config.gmail_sender, config.gmail_receiver, msg.as_string())
        smtp_obj.quit()

        Logger(
            __name__,
            logger_config=LoggerConfig(level=config.log_level),
            filepath=path.join(project_dirpath, config.log_filepath),
        ).info("Finished sending Gmail.")
        print("Finished sending Gmail.")

    @classmethod
    def error(cls, err_title: str, project_dirpath: str, config: WorkflowConfig, critical: bool = False):
        """[classmethod] 現状のワークフローについて何らかのエラーが発生した際にGメールを送信する"""

        if critical:
            subject = "[Critical Error] TriWaveワークフローがシステムエラーにより中断しました"  # 件名
        else:
            subject = "[Error] TriWaveワークフローがエラーにより中断しました"  # 件名

        body = ""  # 本文
        title = "TriWaveにより処理中だったワークフローがエラーにより中断しました"
        project_name = (
            path.abspath(project_dirpath).removeprefix(f"{path.abspath('@')}/")
            if path.abspath(project_dirpath).startswith(f"{path.abspath('@')}/project")
            else path.basename(project_dirpath)
        )

        # ワークフローのHTMLを作成
        metrics = config.metrics.format_for_notifications(config.current_workflow)
        workflow = ""
        for i, w in enumerate(config.workflow):
            workflow += (
                f"<li style='margin-left:1em;'><b>{w['name']}</b>{metrics.workflow_time[i]}</li>"
                if i == config.current_workflow
                else f"<li>{w['name']}{metrics.workflow_time[i]}</li>"
            )

        body = f"""
<html>
    <head></head>
    <body>
        <div>{title}</div>
        <div>確認してください</div><br>
        <div>エラー内容</div>
        <div style='white-space: pre-wrap'>{err_title}</div><br>
        <div>プロジェクト名: <b>{project_name}</b></div>
        <div>実行時間: {metrics.execution_time}</div>
        <div>ワークフロー:</div>
        <ol start='0'>
            {workflow}
        </ol>
    </body>
</html>
        """

        charset = "iso-2022-jp"
        msg = MIMEMultipart()
        msg.attach(MIMEText(body, "html", charset))
        msg["Subject"] = Header(subject.encode(charset), charset)

        with open(path.join(project_dirpath, config.log_filepath), "rb") as f:
            mb = MIMEApplication(f.read())

        mb.add_header("Content-Disposition", "attachment", filename="log.txt")
        msg.attach(mb)

        smtp_obj = smtplib.SMTP("smtp.gmail.com", 587)
        smtp_obj.ehlo()
        smtp_obj.starttls()
        smtp_obj.login(config.gmail_sender, config.gmail_app_pw)

        # メール送信
        Logger(
            __name__,
            logger_config=LoggerConfig(level=config.log_level),
            filepath=path.join(project_dirpath, config.log_filepath),
        ).info("Start sending Gmail.")
        print("Start sending Gmail.")

        smtp_obj.sendmail(config.gmail_sender, config.gmail_receiver, msg.as_string())
        smtp_obj.quit()

        Logger(
            __name__,
            logger_config=LoggerConfig(level=config.log_level),
            filepath=path.join(project_dirpath, config.log_filepath),
        ).info("Finished sending Gmail.")
        print("Finished sending Gmail.")

    @classmethod
    def config_error(
        cls,
        err_msg: list[str],
        project_dirpath: str,
        config: WorkflowConfig | None = None,
    ):
        """[classmethod] 設定ファイルにエラーがある場合にGメールを送信する"""

        subject = "[Config Error] 設定ファイルエラーによりTriWaveが中断しました"
        title = " 設定ファイルエラーによりTriWaveが中断しました"

        body = f"""
<html>
    <head></head>
    <body>
        <div>{title}</div>
        <div>確認してください</div><br>
        <div>エラー内容</div>
        <div style='white-space: pre-wrap'>設定ファイルの読み込みに失敗しました</div>
        <div>{'</div><div>'.join(err_msg)}</div><br>
        <div>プロジェクトディレクトリ: <b>{project_dirpath}</b></div>
    </body>
</html>
        """

        if config is not None:
            gmail_sender = config.gmail_sender
            gmail_app_pw = config.gmail_app_pw
            gmail_receiver = config.gmail_receiver

        else:
            # プロジェクトルートに存在すれば、通知設定ファイルの読み込み
            base_dirpath = path.dirname(path.dirname(path.abspath(__file__)))

            if path.exists(path.join(base_dirpath, "notification.yml")):
                notification_filepath = path.join(base_dirpath, "notification.yml")
            elif path.exists(path.join(base_dirpath, "notification.yaml")):
                notification_filepath = path.join(base_dirpath, "notification.yaml")
            else:
                return False

            with open(notification_filepath, "r", encoding="utf-8") as f:
                notification = yaml.safe_load(f)

            gmail_sender = notification["gmail"]["sender"]
            gmail_app_pw = notification["gmail"]["app_pw"]
            gmail_receiver = notification["gmail"]["receiver"]

        # メール送信
        charset = "iso-2022-jp"
        msg = MIMEMultipart()
        msg.attach(MIMEText(body, "html", charset))
        msg["Subject"] = Header(subject.encode(charset), charset)

        smtp_obj = smtplib.SMTP("smtp.gmail.com", 587)
        smtp_obj.ehlo()
        smtp_obj.starttls()
        smtp_obj.login(gmail_sender, gmail_app_pw)
        print("Start sending Gmail.")

        smtp_obj.sendmail(gmail_sender, gmail_receiver, msg.as_string())
        smtp_obj.quit()
        print("Finished sending Gmail.")

        return True
