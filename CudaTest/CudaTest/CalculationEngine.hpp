#ifndef CalculationEngine_HPP
#define CalculationEngine_HPP

namespace MaybeCuda {
	class CalculationEngine {
	public:

		/// <summary>
		///		The default parameter-less constructor
		/// </summary>
		CalculationEngine();

		/// <summary>
		///		Checks if the device is using CUDA
		/// </summary>
		bool isUsingCuda() { return _usingCuda; }


		/// <summary>
		///		Checks if the device is using CUDA
		/// </summary>
		int CalcVector(bool usingCuda);

		int CalcVector() {
			return this->CalcVector(this->isUsingCuda());
		}

		/// <summary>
		///		The destructor (frees all resources and whatever)
		/// </summary>
		~CalculationEngine() {
			// Free all resources
		}

	private:
		bool _usingCuda = false;

	};
}


#endif // !CalculationEngine_HPP

