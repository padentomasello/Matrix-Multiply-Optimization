################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../benchmark.c \
../sgemm-naive.c \
../sgemm-openmp.c \
../sgemm-small.c 

OBJS += \
./benchmark.o \
./sgemm-naive.o \
./sgemm-openmp.o \
./sgemm-small.o 

C_DEPS += \
./benchmark.d \
./sgemm-naive.d \
./sgemm-openmp.d \
./sgemm-small.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	gcc -std=gnu99 -fopenmp -pg -c -O3 -g3 -Wall -msse4 -fopenmp -pipie -fno-omit-frame-pointer -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


