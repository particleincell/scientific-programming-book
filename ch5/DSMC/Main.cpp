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
	world.addCircle({0.15,0},0.03);
    world.addInlet();

	//set up particle species
    vector<Species> species;
    species.push_back(Species("O", 16*AMU, 0*QE, 1e15, world));
     Species &atoms = species[0];
	
	//setup injection sources
	const double nda = 1e22;			//mean atom density
		
	vector<unique_ptr<Source>> sources;
	sources.emplace_back(new ColdBeamSource(atoms,world,5000,nda));	//ion source

	//setup material interactions
	vector<unique_ptr<Interaction>> interactions;
	interactions.emplace_back(new DSMC_MEX(atoms,world));

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
		world.checkSteadyState(species);


		//screen and file output
        Output::screenOutput(world,species);
        Output::diagOutput(world,species);

		//periodically write out results
        if (world.getTs()%100==0 || world.isLastTimeStep()) {
			Output::fields(world, species);
			Output::particles(world, species,10000);
        }
    }
	
	// show run time
	cout<<"Simulation took "<<world.getWallTime()<<" seconds"<<endl;
	return 0;		//indicate normal exit
}
