#include <math.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <memory>
#include "World.h"
#include "Species.h"
#include "Output.h"
#include "Source.h"
#include "Collisions.h"
#include "PotentialSolver.h"

using namespace std;		//to avoid having to write std::cout
using namespace Const;		//to avoid having to write Const::ME

/*program execution starts here*/
int main(int argc, char *args[])
{
   //initialize domain
    World world(161,121);
    world.setExtents({0,-0.15},{0.4,0.15});
    world.setTime(5e-8,6000);

    //set objects
    double phi_circle = -10;		//set default
    world.addCircle({0.15,0},0.03,phi_circle);
    world.addInlet();

    //set up particle species
    vector<Species> species;
    species.push_back(Species("O", 16*AMU, 0*QE, 1e15, world));
    species.push_back(Species("O+", 16*AMU, 1*QE, 1e2, world));
    Species &atoms = species[0];
    Species &ions = species[1];
	
	//setup injection sources
	const double nda = 1e22;			//mean atom density
	const double ndi = 1e10;			//mean ion density
		
	vector<unique_ptr<Source>> sources;
	sources.emplace_back(new ColdBeamSource(atoms,world,5000,nda));	//ion source
	sources.emplace_back(new ColdBeamSource(ions,world,5000,ndi));	//ion source

	//setup material interactions
	vector<unique_ptr<Interaction>> interactions;
	//interactions.emplace_back(new MCC_CEX(ions,neutrals,world));
	interactions.emplace_back(new DSMC_MEX(atoms,world));

	//initialize potential solver and solve initial potential
    PotentialSolver solver(world,SolverType::GS,5000,1e-4);
    solver.setReferenceValues(0,5,ndi);
    solver.solve();

    //obtain initial electric field
    solver.computeEF();

    // main loop
	while(world.advanceTime()) {

	//inject particles
    	for (auto &source:sources)
    		source->sample();

    	//perform material interactions
    	for (auto &interaction:interactions)  interaction->apply(world.getDt());

		//move particles
		for (Species &sp:species) {
			sp.advance(atoms);
			sp.computeNumberDensity();
			sp.sampleMoments();
			sp.computeMPC();
		}

		// check for steady state
		world.steadyState(species);

		// compute charge density
		world.computeChargeDensity(species);

        //update potential
        solver.solve();

        //obtain electric field
        solver.computeEF();

        /*update averages at steady state*/
        if (world.steadyState(species)) {
        	for (Species &sp:species)
        		sp.updateAverages();
        }

		//screen and file output
        Output::screenOutput(world,species);
        Output::diagOutput(world,species);

	// periodically write out results
        if (world.getTs()%100==0 || world.isLastTimeStep()) {
			Output::fields(world, species);
			Output::particles(world, species,10000);
        }
    }
	
	// show run time
	cout<<"Simulation took "<<world.getWallTime()<<" seconds"<<endl;
	return 0;		//indicate normal exit
}
