#!/usr/bin/env python3
"""
性能监控和调优脚本
用于监控系统性能并提供优化建议
"""
import asyncio
import json
import time
from typing import Dict, Any
import psutil
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import statistics

class PerformanceMonitor:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.metrics = {}

    async def monitor_system_resources(self) -> Dict[str, Any]:
        """监控系统资源使用情况"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / (1024**3),
            "disk_usage": psutil.disk_usage('/').percent,
            "network_connections": len(psutil.net_connections()),
        }

    async def benchmark_api(self, num_requests: int = 100, concurrency: int = 10) -> Dict[str, Any]:
        """API性能基准测试"""
        async def single_request(session: aiohttp.ClientSession, request_id: int):
            start_time = time.time()
            try:
                payload = {
                    "message": f"Test message {request_id}",
                    "thread_id": f"benchmark-{request_id}"
                }
                async with session.post(f"{self.api_url}/chat", json=payload) as response:
                    result = await response.json()
                    end_time = time.time()
                    return {
                        "success": True,
                        "response_time": end_time - start_time,
                        "status_code": response.status
                    }
            except Exception as e:
                end_time = time.time()
                return {
                    "success": False,
                    "response_time": end_time - start_time,
                    "error": str(e)
                }

        response_times = []
        success_count = 0

        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(num_requests):
                tasks.append(single_request(session, i))

            # 分批执行以控制并发
            for i in range(0, len(tasks), concurrency):
                batch = tasks[i:i + concurrency]
                results = await asyncio.gather(*batch, return_exceptions=True)

                for result in results:
                    if isinstance(result, Exception):
                        print(f"Request failed with exception: {result}")
                        continue

                    if result["success"]:
                        success_count += 1
                        response_times.append(result["response_time"])
                    else:
                        print(f"Request failed: {result.get('error', 'Unknown error')}")

        if response_times:
            return {
                "total_requests": num_requests,
                "successful_requests": success_count,
                "success_rate": success_count / num_requests,
                "avg_response_time": statistics.mean(response_times),
                "median_response_time": statistics.median(response_times),
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "p95_response_time": statistics.quantiles(response_times, n=20)[18],  # 95th percentile
                "requests_per_second": len(response_times) / sum(response_times)
            }
        else:
            return {"error": "No successful requests"}

    async def check_health(self) -> Dict[str, Any]:
        """检查服务健康状态"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/health") as response:
                    if response.status == 200:
                        return {"status": "healthy", "response_time": response.headers.get('X-Response-Time')}
                    else:
                        return {"status": "unhealthy", "status_code": response.status}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def generate_report(self, system_metrics: Dict, benchmark_results: Dict, health_status: Dict) -> str:
        """生成性能报告"""
        report = []
        report.append("# 性能监控报告")
        report.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # 系统资源
        report.append("## 系统资源使用情况")
        report.append(f"- CPU使用率: {system_metrics.get('cpu_percent', 'N/A')}%")
        report.append(f"- 内存使用率: {system_metrics.get('memory_percent', 'N/A')}%")
        report.append(f"- 内存使用量: {system_metrics.get('memory_used_gb', 'N/A'):.2f} GB")
        report.append(f"- 磁盘使用率: {system_metrics.get('disk_usage', 'N/A')}%")
        report.append(f"- 网络连接数: {system_metrics.get('network_connections', 'N/A')}")
        report.append("")

        # 服务健康状态
        report.append("## 服务健康状态")
        report.append(f"- 状态: {health_status.get('status', 'unknown')}")
        if 'response_time' in health_status:
            report.append(f"- 响应时间: {health_status['response_time']}")
        report.append("")

        # 性能基准测试
        if "error" not in benchmark_results:
            report.append("## API性能基准测试")
            report.append(f"- 总请求数: {benchmark_results['total_requests']}")
            report.append(f"- 成功请求数: {benchmark_results['successful_requests']}")
            report.append(f"- 成功率: {benchmark_results['success_rate']:.2%}")
            report.append(f"- 平均响应时间: {benchmark_results['avg_response_time']:.3f}s")
            report.append(f"- 中位数响应时间: {benchmark_results['median_response_time']:.3f}s")
            report.append(f"- 95%响应时间: {benchmark_results['p95_response_time']:.3f}s")
            report.append(f"- QPS (每秒请求数): {benchmark_results['requests_per_second']:.2f}")
            report.append("")

            # 性能评估
            report.append("## 性能评估")
            if benchmark_results['avg_response_time'] < 1.0 and benchmark_results['success_rate'] > 0.95:
                report.append("✅ 性能良好")
            elif benchmark_results['avg_response_time'] < 3.0 and benchmark_results['success_rate'] > 0.90:
                report.append("⚠️ 性能一般，建议优化")
            else:
                report.append("❌ 性能不佳，需要紧急优化")

            if benchmark_results['requests_per_second'] > 50:
                report.append("✅ 高并发处理能力良好")
            elif benchmark_results['requests_per_second'] > 20:
                report.append("⚠️ 中等并发处理能力")
            else:
                report.append("❌ 并发处理能力不足")
        else:
            report.append("## 基准测试失败")
            report.append(f"错误: {benchmark_results['error']}")

        return "\n".join(report)

async def main():
    monitor = PerformanceMonitor()

    print("开始性能监控...")

    # 收集系统指标
    print("收集系统资源信息...")
    system_metrics = await monitor.monitor_system_resources()

    # 检查服务健康状态
    print("检查服务健康状态...")
    health_status = await monitor.check_health()

    # 运行基准测试
    print("运行API基准测试...")
    benchmark_results = await monitor.benchmark_api(num_requests=50, concurrency=5)

    # 生成报告
    report = monitor.generate_report(system_metrics, benchmark_results, health_status)

    print("\n" + "="*50)
    print(report)
    print("="*50)

    # 保存报告到文件
    with open("performance_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("报告已保存到 performance_report.md")

if __name__ == "__main__":
    asyncio.run(main())