
#include <stdio.h>
#include "CalculationEngine.hpp"

int main()
{
	printf("Running Calculations...\n\n");
	MaybeCuda::CalculationEngine engine = MaybeCuda::CalculationEngine();

	int val = engine.CalcVector(false);
	printf("Result of CPU Calculation = %d\n", val);
	if (engine.isUsingCuda()) {
		val = engine.CalcVector(true);
		printf("Result of GPU Calculation = %d\n", val);
	}
	else {
		printf("GPU driver not available for comparison\n");
	}
	printf("\n\n\nHit enter to exit\n");
	getchar();
	return 0;
}